@echo off
echo GitHub'a yukleniyor...
echo.

:: Tum degisiklikleri ekle
git add .

:: Tarih ve saat ile otomatik commit mesaji olustur
set datetime=%date% %time%
git commit -m "Otomatik Guncelleme: %datetime%"

:: Uzak sunucuya gonder
git push

echo.
if %errorlevel% equ 0 (
    echo [BASARILI] Kodlar GitHub'a gonderildi.
) else (
    echo [HATA] Bir sorun olustu. Lutfen yukaridaki hatalari kontrol edin.
)
echo.
pause
