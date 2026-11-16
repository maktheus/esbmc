# Banner digital — XXXV CONIC 25/26

- `banner_conic_pibiti.png` — 1080×1920 px, RGB, ~300 KB (limite: 5 MB).
- `banner.html` — fonte editável (Arial/Helvetica; margem de segurança de 60 px).
- `render.js` — regera o PNG: `node render.js` (requer Playwright).

## Falta só o QR Code (obrigatório)
1. Suba o **resumo expandido** (PDF) no Google Drive → "Qualquer pessoa com o link" → Leitor.
2. Gere o QR no Canva (https://www.canva.com/pt_br/gerador-qr-code/) com esse link.
3. Salve a imagem como `qr.png` nesta pasta e rode `node render.js`.
4. Teste com a câmera do celular (de outra conta Google) antes de enviar.

## Conferir antes de enviar
- Substituir pelo modelo oficial se ele sair antes do prazo (18/11), mantendo o conteúdo.
- Logo da unidade acadêmica é opcional (rodapé).
- Todos os números vêm de `pibic/relatorio_final_2026/secoes/` (rodada canônica, `RandomState(42)`).
