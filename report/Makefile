main.pdf: main.tex sections/*.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	@if grep -q '\\bibdata' main.aux; then bibtex main; fi
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

check: main.pdf
	@pages=$$(pdfinfo main.pdf | awk '/^Pages:/ {print $$2}'); \
	if grep -q '^\\coverpagetrue' main.tex; then content=$$((pages-1)); else content=$$pages; fi; \
	echo "$$pages pages in total, $$content counted against the 8-page limit"; \
	if grep -q "undefined" main.log; then echo "main.log still has undefined references"; exit 1; fi; \
	[ $$content -le 8 ]

clean:
	rm -f main.aux main.log main.out main.toc main.bbl main.blg main.fls main.synctex.gz

.PHONY: clean check
