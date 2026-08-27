# pdf2md

Menggabungkan dua model terpisah —
Unlimited-OCR untuk membaca layout dokumen, dan sebuah vision LLM untuk menjelaskan
isi setiap gambar yang tidak dimengerti model OCR.

Dua antarmuka di atas pipeline yang sama: webapp Streamlit (`app.py`) dan HTTP API
(`pdf2md/api.py`) untuk `kirim PDF → terima .md`.

## Pipeline

| # | Tahap | Isi |
|---|-------|-----|
| 1 | Render halaman PDF | PyMuPDF merender tiap halaman jadi PNG sesuai DPI |
| 2 | OCR layout | tiap halaman dikirim ke Unlimited-OCR; blok bergrounding `<\|det\|>` jadi markdown, region gambar/chart dipotong jadi PNG |
| 3 | Deskripsi gambar | tiap potongan gambar dikirim ke vision LLM bersama caption-nya sebagai konteks; gambar yang memuat tabel di-OCR jadi tabel markdown GFM, bukan diringkas |
| 4 | Substitusi placeholder | `![chart](figures/...)` diganti deskripsi hasil tahap 3 |
| 5 | Hapus header/footer | baris yang berulang di tepi tiap halaman dan nomor halaman dibuang otomatis |

### Tahap 2 dan 3 tumpang-tindih

Tahap 3 tidak menunggu seluruh OCR selesai. Begitu satu halaman selesai diparsing
dan gambarnya dipotong, gambar itu langsung masuk antrean vision — jadi deskripsi
gambar halaman 1 sudah dikerjakan sementara halaman-halaman terakhir masih di-OCR.

Keduanya juga paralel di dalam tahapnya sendiri: `OCR_CONCURRENCY` halaman sekaligus,
`VISION_CONCURRENCY` gambar sekaligus. Untuk dokumen yang gambarnya menumpuk di awal,
hampir seluruh latensi vision tersembunyi di balik OCR yang masih berjalan.

Konsekuensinya di UI: penyebut tahap 3 bertambah selama gambar baru ditemukan, jadi
ditampilkan sebagai `0/2+` — tanda `+` berarti angkanya belum final. Bar progres
memakai bobot per tahap, bukan "tahap ke-N selesai", supaya tidak pernah mundur saat
tick OCR dan vision datang bergantian.

Satu halaman atau satu gambar yang gagal tidak menggagalkan seluruh run: halaman
gagal ditandai komentar HTML, gambar gagal tetap menyisakan caption plus alasan
gagalnya.

## Jalankan

```bash
cd pdf2md
cp .env.example .env      # isi endpoint & API key

uv run streamlit run app.py                                  # UI
uv run uvicorn pdf2md.api:app --host 0.0.0.0 --port 8080     # API
```

`uv` mengurus venv dan dependensi sendiri; tidak perlu `uv sync` manual.
Streamlit ada di dependency group `ui`, jadi image API bisa dipasang tanpanya.

## Docker

`Dockerfile` mengemas **API-nya saja** (`pdf2md.api:app` di balik uvicorn); UI
Streamlit tidak ikut. Konfigurasi seluruhnya dari environment: tidak ada `.env` di
dalam image (`.dockerignore` memblokirnya), key dikirim saat run.

```bash
docker compose up --build                 # baca .env, terbit di :8080
curl -F file=@paper.pdf http://127.0.0.1:8080/convert -o paper.md
```

Tanpa compose:

```bash
docker build -t pdf2md-api .
docker run --rm -p 8080:8080 --env-file .env \
  --read-only --tmpfs /tmp:size=2g,mode=1777 pdf2md-api
```

### Endpoint di host, bukan di container

Ini kesalahan paling sering: di dalam container, `127.0.0.1` adalah container itu
sendiri, bukan laptopmu. Kalau `OCR_BASE_URL` di `.env` menunjuk `http://127.0.0.1:8000`,
timpa saat run:

```bash
-e OCR_BASE_URL=http://host.docker.internal:8000
```

Di Docker Desktop alias itu sudah ada. Di host Linux, aktifkan `extra_hosts` yang
sudah disiapkan (dikomentari) di `compose.yaml`. Endpoint publik (`https://...`)
tidak butuh apa pun.

### Isi container

Render halaman dan upload masuk ke `TMPDIR` (`/tmp`), lalu dihapus tiap request
selesai. `compose.yaml` menjalankan rootfs `read_only` dengan `/tmp` sebagai tmpfs
2 GB, jadi tidak ada yang ditulis ke layer container — halaman 300 DPI cukup besar
dan umurnya pendek. Kalau PDF-mu besar dan `OCR_DPI` tinggi, naikkan ukuran tmpfs
itu. Proses jalan sebagai user `pdf2md` (uid 10001), bukan root, dan `HEALTHCHECK`
menembak `/health` (tidak butuh auth, jadi tetap jalan meski `API_KEY` diisi).

Satu worker uvicorn, disengaja: tiap request sudah memakai `OCR_CONCURRENCY` +
`VISION_CONCURRENCY` thread dan `API_MAX_CONCURRENT` sudah membatasi run paralel di
dalam proses. Untuk kapasitas lebih, tambah replica, bukan `--workers`.

## API

```bash
curl -F file=@paper.pdf http://127.0.0.1:8080/convert -o paper.md
```

Itu saja untuk kasus dasar: kirim PDF, dapat `.md`. Dokumentasi interaktif ada di
`/docs`, dan `/health` menampilkan konfigurasi model yang sedang aktif (tanpa API key).

### `POST /convert`

| Bagian | Isi |
|--------|-----|
| body | `multipart/form-data`, field `file` berisi PDF |
| `?format=` | `md` (default) → dokumen sebagai attachment `text/markdown`; `json` → Markdown plus manifest gambar |
| `?dpi=` | timpa `OCR_DPI` untuk run ini (72–600) |
| `?cleanup=` | timpa `CLEANUP_ENABLED` |
| `?keep_image_link=` | timpa `KEEP_IMAGE_LINK` |
| header | `X-API-Key` bila `API_KEY` diisi di `.env` |

Pada `format=md`, angka hasil run ikut sebagai header: `X-Pdf2md-Pages`,
`X-Pdf2md-Figures`, `X-Pdf2md-Described`, `X-Pdf2md-Substituted`,
`X-Pdf2md-Chrome-Removed`, `X-Pdf2md-Page-Failures`. `format=json` memberi angka yang
sama sebagai field, ditambah `figures[]` (halaman, kategori, caption, deskripsi, error)
dan daftar halaman yang gagal di-OCR.

Kode status: `400` PDF kosong/tidak bisa dirender, `413` melewati
`API_MAX_UPLOAD_MB`, `415` bukan PDF, `401` API key salah, `502` semua halaman gagal
di-OCR (endpoint OCR yang bermasalah, bukan filenya).

Endpoint dan API key model dibaca sekali dari `.env` saat start, jadi tidak ada
setelan per request selain tiga override di atas. Satu proses memproses
`API_MAX_CONCURRENT` PDF sekaligus; sisanya mengantre, supaya request berbarengan
tidak mengalikan `OCR_CONCURRENCY` ke endpoint OCR.

Direktori kerja tiap request bersifat temporer dan langsung dihapus setelah respons
disusun: Markdown dan manifest dikirim sebagai nilai, tidak ada file yang dilayani
dari disk. Yang butuh file crop-nya harus pakai UI atau `run_pipeline` langsung.

## Setelan `.env`

`cp .env.example .env`, lalu isi. Nilai model bisa ditimpa dari sidebar Streamlit
per run; di API, `.env` dibaca **sekali saat start** dan yang bisa ditimpa per
request hanya `dpi`, `cleanup`, `keep_image_link`.

Yang wajib diisi cuma tiga: `OCR_BASE_URL`, `VISION_BASE_URL`, `VISION_API_KEY`.
Sisanya punya default yang wajar. Setiap baris kosong (`OCR_MODEL=`) jatuh ke
default, bukan jadi string kosong, dan angka yang tidak bisa diparsing juga jatuh
ke default alih-alih membuat server gagal start.

### Stage 1 — Unlimited-OCR

| Variabel | Default | Isi |
|----------|---------|-----|
| `OCR_BASE_URL` | `http://127.0.0.1:8000` | Root endpoint vLLM; klien menambahkan `/v1/chat/completions` sendiri. **Tanpa** `/v1`. |
| `OCR_API_KEY` | kosong | Hanya kalau vLLM-mu dijalankan dengan `--api-key`. |
| `OCR_MODEL` | `baidu/Unlimited-OCR` | Harus sama dengan nama model yang diserve. |
| `OCR_DPI` | `300` | Resolusi render halaman. Turunkan untuk hemat waktu/token, naikkan untuk scan buruk. |
| `OCR_MAX_TOKENS` | `8192` | Lihat catatan di bawah; jangan disetel ke `max_model_len`. |
| `OCR_TIMEOUT` | `1800` | Detik per halaman. |
| `OCR_CONCURRENCY` | `4` | Halaman diproses sekaligus. Batasi sesuai kapasitas GPU-mu. |
| `OCR_RETRIES` | `4` | Percobaan per halaman; error 4xx tidak diulang. |
| `OCR_FIGURE_PAD` | `6` | Piksel margin saat memotong gambar. |

### Stage 2 — vision LLM

| Variabel | Default | Isi |
|----------|---------|-----|
| `VISION_BASE_URL` | `https://api.openai.com/v1` | Root `/v1`; SDK menambahkan `/chat/completions`. **Pakai** `/v1` di sini — beda dari OCR. |
| `VISION_API_KEY` | kosong | Key endpoint vision. |
| `VISION_MODEL` | `gpt-4o` | Model apa pun yang menerima input gambar. |
| `VISION_TIMEOUT` | `300` | Detik per gambar. |
| `VISION_TEMPERATURE` | `0.2` | Rendah supaya deskripsi tidak mengarang. |
| `VISION_MAX_TOKENS` | `600` | Naikkan kalau gambarmu banyak memuat tabel. |
| `VISION_CONCURRENCY` | `4` | Gambar dideskripsikan sekaligus; perhatikan rate limit. |
| `VISION_RETRIES` | `3` | Percobaan per gambar. |
| `VISION_PROMPT` | prompt bawaan (Indonesia) | Kosongkan untuk memakai bawaan. |

### Stage 4 — header/footer dan output

| Variabel | Default | Isi |
|----------|---------|-----|
| `CLEANUP_ENABLED` | `1` | Matikan (`0`) kalau pembersihan memakan isi dokumen. |
| `CLEANUP_MIN_RATIO` | `0.6` | Fraksi halaman yang harus memuat satu baris agar dianggap chrome. |
| `CLEANUP_MIN_PAGES` | `3` | Dokumen lebih pendek dari ini tidak dibersihkan secara statistik. |
| `CLEANUP_ZONE_LINES` | `3` | Berapa baris teratas/terbawah tiap halaman yang boleh dibuang. |
| `CLEANUP_DROP_PAGE_NUMBERS` | `1` | Buang nomor halaman walau tidak berulang verbatim. |
| `KEEP_IMAGE_LINK` | `1` | `1` menyimpan link gambar di samping deskripsi, `0` mengganti placeholder dengan deskripsi saja. |

### HTTP API

| Variabel | Default | Isi |
|----------|---------|-----|
| `API_KEY` | kosong | **Kosong berarti server terbuka tanpa autentikasi.** Isi untuk mewajibkan header `X-API-Key`. |
| `API_MAX_UPLOAD_MB` | `50` | Upload lebih besar ditolak `413` sambil dibaca, bukan setelah. |
| `API_MAX_CONCURRENT` | `2` | PDF diproses sekaligus; sisanya mengantre. |

Nilai bool menerima `1/0`, `true/false`, `yes/no`, `on/off`.

`.env` berisi API key: sudah masuk `.gitignore` dan `.dockerignore`, jadi tidak
pernah ikut ke image. Di container, isinya masuk lewat `--env-file` / `env_file`,
bukan lewat file di dalam image.

### Contoh minimal

```dotenv
# OCR jalan di host yang sama
OCR_BASE_URL=http://127.0.0.1:8000
OCR_MODEL=baidu/Unlimited-OCR

# Vision di layanan OpenAI-compatible
VISION_BASE_URL=https://api.openai.com/v1
VISION_API_KEY=sk-...
VISION_MODEL=gpt-4o

# Server terbuka? isi ini.
API_KEY=ganti-aku
```

Di Docker, `OCR_BASE_URL=http://127.0.0.1:8000` menunjuk ke container itu sendiri;
pakai `http://host.docker.internal:8000`. Lihat bagian Docker di atas.

### Catatan penting soal `OCR_MAX_TOKENS`

`max_tokens` + token prompt harus tetap di bawah `max_model_len` (32768). Satu
halaman A4 pada 300 DPI sudah memakan ~2.7k token prompt, jadi menyetel 32768 di
sini akan ditolak server dengan HTTP 400. Default 8192 aman.

## Output

Hasil run ditulis ke direktori temporer:

- `output.md` — dokumen final, sama dengan yang tampil di tab **Markdown (copy)**.
- `figures/` — tiap gambar hasil crop.
- `figures.json` — manifest gambar (halaman, kategori, box, caption) plus deskripsi
  yang didapat masing-masing, untuk menelusuri hasil yang mencurigakan.

Di UI, tab **Markdown (copy)** menyediakan tombol copy satu klik, dan ada tombol
unduh `output.md`.

## Deteksi header/footer

Dua kekuatan pencocokan, karena risiko false-positive-nya berbeda:

- **exact** — baris berulang verbatim. Aman di seluruh zona tepi halaman.
- **folded** — baris berulang setelah deretan angka diabaikan, yang menangkap
  footer macam `Page 7 of 14`. Karena pelipatan angka juga menyatukan baris isi
  yang sebenarnya berbeda (`Bab 1`, `Bab 2`), pencocokan ini hanya dipercaya pada
  baris konten paling atas atau paling bawah halaman.

Baris yang ditandai model sendiri sebagai `header`/`footer`/`page_number` langsung
dipercaya tanpa ambang pengulangan. Gambar, tabel, heading, rumus, dan list
dikecualikan dari pembersihan. Dokumen di bawah `CLEANUP_MIN_PAGES` halaman tidak
dibersihkan secara statistik: pengulangan belum bisa dibedakan dari isi.

Panel **Header/footer yang dihapus** di UI menampilkan tepat apa yang dibuang, jadi
pembersihan yang kelewat agresif langsung kelihatan.

## Tes

```bash
uv run pytest
```
