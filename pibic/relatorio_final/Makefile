# Relatorio Final PIBIC 2025-2026
# Uso: make | make clean | make distclean

DOC = relatorio_final

.PHONY: all clean distclean

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex secoes/*.tex apendices/*.tex referencias.bib
	pdflatex -interaction=nonstopmode $(DOC).tex
	-bibtex $(DOC)
	pdflatex -interaction=nonstopmode $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex
	@echo "--- $(DOC).pdf gerado ---"

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).lof $(DOC).lot \
	      $(DOC).bbl $(DOC).blg $(DOC).brf $(DOC).synctex.gz

distclean: clean
	rm -f $(DOC).pdf
