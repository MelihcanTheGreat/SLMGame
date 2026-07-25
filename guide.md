# Oyun Kılavuzu (Game Guide)

Bu kılavuz, oyuncuların NPC'lere verebileceği geçerli komutları tanımlar.

## Geçerli Aksiyonlar (action_type)
- `move`: Belirli bir yere gitme veya bir hedefe yaklaşma.
- `gather`: Kaynak toplama (odun, taş vb.).
- `attack`: Bir düşmana veya hedefe saldırma.
- `defend`: Bulunulan konumu veya bir hedefi savunma.
- `idle`: Bekleme, hiçbir şey yapmama veya işlemi iptal etme.
- `talk`: Konuş oyuncu ile.

## Hedefler (target_object)
- Kaynaklar: `wood` (odun), `stone` (taş)
- Düşmanlar/Karakterler: `enemy_name` (örneğin goblin, ork), veya diğer spesifik hedefler.
- Hedef belirtilmemişse: `null`

## Konum (target_location)
- x, y, z koordinatları olarak belirtilir.
- Eğer oyuncu koordinat belirtmemişse `is_specified` değeri `false` olmalıdır.
