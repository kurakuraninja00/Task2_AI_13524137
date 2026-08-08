## PERAN

Kamu adalah asisten laboratorium yang bertugas menyusun **dokumen spesifikasi Tugas Besar Inteligensi Artifisial** dengan topik **Local Search**, menggunakan studi kasus permainan **N-Queens**. Dokumen ini akan dijadikan **acuan implementasi** oleh mahasiswa lain, sehingga harus lengkap, teknis, konsisten, dan tidak ambigu.

## FORMAT OUTPUT

- Tulis **seluruh dokumen dalam format LaTeX** (kode `.tex` lengkap, siap dikompilasi). Gunakan package standar sesuai kebutuhan (misalnya `amsmath`, `graphicx`, `hyperref`, `listings`/`minted` untuk cuplikan kode, `enumitem` untuk daftar).
- **Sisakan/khususkan halaman pertama sebagai halaman judul**, memuat minimal:
  - Judul dokumen: "Spesifikasi Tugas Besar Inteligensi Artifisial — Local Search: N-Queens Problem"
  - Placeholder Nama: `[NAMA MAHASISWA]`
  - Placeholder NIM: `[NIM MAHASISWA]`
- Gunakan `\newpage`/`\clearpage` setelah halaman judul agar tidak bercampur dengan isi lain.
- Sertakan **Daftar Isi** (table of contents) setelah halaman judul.
- Gunakan heading terstruktur (`\section`, `\subsection`, dst.) untuk setiap bagian.

## TEMA & BATASAN ESENSIAL

- Permasalahan yang dirancang adalah **N-Queens**, dan permasalahan ini **harus dapat diformulasikan sebagai persoalan Local Search**, yaitu berbasis *state* lengkap (*complete-state formulation*) **tanpa** memerlukan *path* menuju goal.
- Dokumen yang dihasilkan adalah **dokumen spesifikasi**, bukan laporan hasil eksperimen — fokus pada rancangan, formulasi, dan penjelasan, bukan pada hasil run.

## ISI MINIMAL DOKUMEN SPESIFIKASI (WAJIB — jangan ada yang terlewat)

Susun bagian spesifikasi N-Queens dengan memuat seluruh poin berikut secara eksplisit dan berurutan:

1. **Tujuan dan deskripsi permasalahan** — jelaskan N-Queens sebagai persoalan local search.
2. **Representasi state** sebagai konfigurasi lengkap (*complete-state formulation*) — misalnya representasi posisi seluruh ratu pada papan N×N.
3. **Aturan serta batasan (constraints)** permasalahan (misalnya larangan saling menyerang antar ratu, ukuran papan N, dsb).
4. **Formulasi initial state** — dibuat secara **random**.
5. **Successor function** dan langkah **neighbor (move)** yang diperbolehkan pada tiap iterasi.
6. **Objective function atau heuristic cost function** — bebas dirancang, minimal 1, **sertakan penjelasan alasan pemilihannya** (misalnya jumlah pasangan ratu yang saling menyerang).
7. **Daftar fungsi atau command utama** beserta penjelasan singkat masing-masing.
8. **Contoh input dan output** yang konkret (misalnya contoh representasi state awal random dan state akhir hasil pencarian).

## KONSEP IMPLEMENTASI YANG WAJIB DIJELASKAN DALAM DOKUMEN

Spesifikasi harus menjelaskan rancangan agar implementasi nantinya mampu menerapkan:

- Varian **Hill-Climbing**: Basic (Steepest-Ascent), Sideways Move, Stochastic, dan Random Restart — jelaskan cara kerja tiap varian pada konteks N-Queens
- **Simulated Annealing** — termasuk skema *cooling schedule* dan probabilitas penerimaan state yang lebih buruk.
- **Genetic Algorithm** — jelaskan representasi kromosom N-Queens, populasi, *selection*, *crossover*, dan *mutation*.
- **Representasi state dan successor/neighbor function** (konsisten dengan bagian spesifikasi di atas).
- **Objective/heuristic function** (konsisten dengan bagian spesifikasi di atas).
- **Tambahkan visualisasi atau animasi proses pencarian**, misalnya perubahan *state* antar iterasi (contoh: visualisasi papan N-Queens tiap iterasi) atau grafik nilai *objective function* yang berubah seiring iterasi berjalan. 

## ATURAN DAN LARANGAN YANG HARUS DIPATUHI

Cantumkan/patuhi ketentuan berikut sebagai bagian dari konteks dokumen (bisa dalam bentuk catatan/preamble di awal dokumen):

- gunakan struktur heading, penomoran section, dan penamaan yang rapi dan mudah dirujuk.

## GAYA PENULISAN

- Gunakan Bahasa Indonesia formal/teknis, konsisten dengan istilah pada dokumen acuan (*state*, *neighbor*, *successor function*, *objective function*, *heuristic*, dsb. — istilah teknis boleh tetap dalam Bahasa Inggris dan dicetak miring).
- Struktur dokumen sistematis dengan heading/subheading LaTeX (`\section`, `\subsection`).
- Sertakan contoh konkret (misalnya representasi array posisi ratu, contoh *initial state* random pada papan 8×8, dsb.) agar dokumen benar-benar dapat dijadikan acuan implementasi.

## OUTPUT YANG DIHARAPKAN

Hasilkan **satu file `.tex` utuh dan siap dikompilasi**, mencakup halaman judul (dengan placeholder Nama & NIM), daftar isi, seluruh isi minimal spesifikasi, penjelasan konsep implementasi wajib seluruhnya bertema **N-Queens**