# pdf2md

PDF → Markdown siap copy. Menggabungkan dua model yang tadinya terpisah —
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

## Setelan

Semua nilai di `.env` (lihat `.env.example`) bisa ditimpa dari sidebar Streamlit per
run, jadi `.env` hanya menyediakan default. Di API, `.env` dibaca sekali saat start;
yang bisa ditimpa per request hanya `dpi`, `cleanup`, dan `keep_image_link`.
Yang paling sering diubah:

- `OCR_BASE_URL` / `OCR_MODEL` — endpoint vLLM Unlimited-OCR.
- `VISION_BASE_URL` / `VISION_MODEL` / `VISION_API_KEY` — endpoint vision
  OpenAI-compatible. Pakai root `/v1`; SDK menambahkan `/chat/completions`.
- `VISION_PROMPT` — prompt deskripsi gambar. Kosongkan untuk memakai prompt bawaan.
- `KEEP_IMAGE_LINK` — `1` menyimpan link gambar di samping deskripsi, `0` mengganti
  placeholder dengan deskripsi saja.
- `API_KEY` — kosong berarti API terbuka tanpa autentikasi; isi untuk mewajibkan
  header `X-API-Key`. `API_MAX_UPLOAD_MB` dan `API_MAX_CONCURRENT` membatasi ukuran
  upload dan jumlah run berbarengan.

`.env` berisi API key dan sudah masuk `.gitignore`.

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
