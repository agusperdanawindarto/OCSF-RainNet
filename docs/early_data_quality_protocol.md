# NASA GPM IMERG V07B Final — Akuisisi dan Verifikasi Label Rainfall

**Pembaruan:** 17 November 2026  
**Root:** `D:\DISERTASI_S3\SEGMENTASI_HUJAN`  
**Tujuan:** memperoleh **estimasi numerik curah hujan** NASA IMERG V07B dalam **mm/jam** yang sesuai dengan citra satelit Himawari, bukan mengubah warna produk visual JMA menjadi label yang tidak terukur.

## Status aktual berdasarkan alat langsung

- Perangkat Windows `DESKTOP-FVE5NQR`: online.
- Drive D: ruang bebas sekitar 239,7 GB.
- Indeks GES DISC CMR `GPM_3IMERGHH`, V07: **48 granula** untuk 2023-09-20 dan **48 granula** untuk 2024-01-07.
- Arsip 2023 memuat **140 timestamp B13 unik**; arsip 2024-01-07 memuat **3 timestamp B13**.
- Manifest ini memilih **7 granula** dengan perbedaan maksimum 5 menit terhadap pusat interval IMERG 30 menit.
- NASA HTTPS tanpa autentikasi mengembalikan **HTTP 302** ke `urs.earthdata.nasa.gov`.
- File `.HDF5` IMERG lokal: **0**, sehingga label intensitas hujan **belum tersedia**.
- Dua granula 2023 merupakan pilot pertama; kemudian perluas menjadi semua 7 setelah QC.

## 1. Provenance yang wajib dipertahankan

| Jenis | Sumber | Status |
|---|---|---|
| Himawari B13 | JMA Himawari; arsip eksperimen RainNet | Citra dan SHA-256 lokal tersedia |
| NASA CMR metadata | `cmr.earthdata.nasa.gov` | Sudah diperoleh dan diindeks |
| NASA GPM 3IMERGHH V07B HDF5 | `data.gesdisc.earthdata.nasa.gov` | Unduh membutuhkan Earthdata |
| Curah hujan hasil mask multikelas | `Grid/precipitation` pada HDF5 V07B | Belum dibuat, menunggu data asli |
| Validasi BMKG ARG/AWS/radar | Observasi independen | Belum tersedia |

**Penting:** citra Himawari historis dari tanggal 2023 bisa pernah dipakai untuk melatih RainNet asli; tanggal ini hanya boleh digunakan sebagai pilot/riset tambahan. Evaluasi independen selanjutnya harus menggunakan hari atau kejadian meteorologis baru yang tidak tumpang tindih.

## 2. Variabel resmi V07B

NASA IMERG V07 mengganti variabel V06 `Grid/precipitationCal` menjadi **`Grid/precipitation`**. Satuannya `mm/hr`, bukan akumulasi 30 menit (`mm`). File input yang dipakai adalah *Final Run* `GPM_3IMERGHH` V07B, dengan resolusi geografis sekitar 0,1° dan interval 30 menit. Data native HDF5 berbentuk `(time=1, lon=3600, lat=1800)`; loader perlu transpose saat membuat peta utara–selatan.

**NASA FAQ 24 Februari 2025:** https://gpm.nasa.gov/sites/default/files/2025-02/FAQ%20for%20IMERG%20V07%20Applications%20Users_202502.pdf  
**Dataset NASA:** https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary  
**Registrasi Earthdata:** https://urs.earthdata.nasa.gov/

## 3. Manifest pilot yang sudah diseleksi

`metadata/cmr/IMERG_V07_HIMAWARI_PILOT_MANIFEST.json` mencatat URL resmi NASA, nama granula, stempel UTC, lokasi B13 terkait, perbedaan waktu, serta lokasi file output **drive D**.

| Tanggal UTC | Interval IMERG (mulai UTC) | B13 terpilih (UTC) | Selisih dari titik tengah |
|---|---|---|---:|
| 2023-09-20 | 14.00 | 14.10 | 5 menit |
| 2023-09-20 | 14.30 | 14.50 | 5 menit |
| 2023-09-20 | 15.00 | 15.10 | 5 menit |
| 2023-09-20 | 15.30 | 15.40 | 5 menit |
| 2023-09-20 | 16.00 | 16.10 | 5 menit |
| 2023-09-20 | 16.30 | 16.40 | 5 menit |
| 2024-01-07 | 14.00 | 14.10 | 5 menit |

Ini merupakan **pencocokan kandidat berdasarkan nama berkas/UTC**, belum validasi penuh radiometri dan registrasi tiap gambar.

## 4. Urutan notebook yang dianjurkan

Buka `06_IMERG_V07_Acquisition_and_QC.ipynb` di VS Code menggunakan Python 3.8.10:

1. Jalankan sel parameter; `DOWNLOAD_NOW=False` menjaga unduhan nonaktif.
2. Periksa tujuh kandidat serta lokasi D.
3. Login melalui halaman resmi NASA Earthdata (akun gratis).
4. Untuk mengunduh **dua granula pilot** melalui notebook, ubah `DOWNLOAD_NOW=True` dan isi kredensial melalui input **lokal** yang tersembunyi. **Jangan menaruh password di chat, .py, .ipynb, .env, README atau GitHub.**
5. File HDF5 ditulis streaming ke `.part`, diverifikasi, baru dipindahkan ke lokasi tujuan. Berkas rusak/HTML/login ditolak; tidak ada label palsu.
6. Jalankan QC dan lihat laporan JSON/CSV `metadata/inventories/IMERG_V07_PILOT_QC`.
7. Jika semua berhasil, lanjutkan `02_IMERG_Himawari_Alignment_and_Multiclass.ipynb` dan audit mask intensitas (bukan training dulu).

Alternatif perintah terminal VS Code:

```text
py -3.8 scripts/prepare_imerg_pilot_manifest.py
py -3.8 scripts/download_imerg_pilot_v07.py --limit 2
py -3.8 scripts/verify_imerg_v07.py
py -3.8 scripts/test_imerg_pipeline.py
```

Perintah di atas aman (*dry-run*). Unduhan nyata hanya dari notebook dengan autentikasi, atau dari terminal privat yang telah menyediakan kredensial NASA di environment proses:

```text
py -3.8 scripts/download_imerg_pilot_v07.py --download --limit 2
```

Gunakan `--all --download` untuk seluruh manifest **setelah pilot lolos**, bukan sebelum.

## 5. QA gate wajib sebelum label digunakan

File lolos verifikasi apabila:

1. Header berupa signature HDF5 yang benar, bukan HTTP 200 berisi halaman login.
2. Nama file memuat `3B-HHR.MS.MRG.3IMERG`, tanggal dan versi `V07B`; cocok dengan CMR.
3. `Grid/precipitation` ditemukan, satuannya benar **mm/hr**.
4. `Grid/lat`, `Grid/lon` dan bentuk `(1,3600,1800)` benar.
5. Indonesia dapat dipotong dengan bbox 95–141 BT dan -11,5 hingga 7 LU tanpa kesalahan orientasi.
6. No-data dibuang, nilai tak terhingga/negatif tidak dianggap curah hujan.
7. Minimal 95% piksel potongan valid dan laju hujan berada dalam rentang yang wajar bagi IMERG V07 (peringatan terhadap nilai ekstrem).
8. SHA-256 dan ringkasan kelas disimpan bersama waktu dan bukti asal.
9. Kelas multikelas baru dibentuk dari **nilai mm/jam**: 0: [0,1), 1: [1,5), 2: [5,10), 3: [10,20], 4: >20, 255: invalid.
10. Dataset sebelum training harus memiliki hari/kejadian berbeda dan kelas jarang yang memadai; **QC satu file tidak membuktikan kesiapan training**.

## 6. Hasil uji kode (tidak sama dengan hasil penelitian)

- Tujuh pasangan kandidat lolos pembatasan selisih waktu (<= 5,01 menit).
- Validator HDF5 berhasil membaca fixture sementara dengan grid V07 yang sesuai.
- Validator menolak fixture saat satuan diubah menjadi `mm/day`.
- Fixture sintetis **dihapus** setelah pengujian, **tidak** pernah dimasukkan ke raw/processed.
- Pemeriksaan aktual melaporkan **0 verified; 7 missing** karena autentikasi Earthdata belum tersedia.

## 7. Validitas ilmiah untuk Penakluk_Top_Tier

Pemetaan kelas berdasarkan IMERG merupakan **supervisi label estimasi curah hujan multisatelit**, bukan pengukuran BMKG yang independen. Hasilnya harus dilaporkan sebagai `IMERG-consistent satellite rainfall-intensity segmentation` hingga ada data referensi radar/ARG/AWS. Citra HRP/SND hanya visualisasi meteorologi; warna tidak menggantikan variabel intensitas mm/hr.

Tidak ada training, data synthetis untuk eksperimen, atau klaim peningkatan OCSF-RainNet yang dilakukan dalam fase persiapan ini.
