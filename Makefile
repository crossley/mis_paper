# simple Makefile to produce stac.pdf
TEX = pdflatex # typesetting engine
MAIN = main
TEXFILES = $(wildcard *.tex)
SECTIONS = $(wildcard sections/*.tex)
BIBFILES = $(wildcard bibs/*.bib)
FIGS = $(wildcard figs/*)

.PHONY: all clean

all: ${MAIN}.pdf

${MAIN}.pdf: ${TEXFILES} ${SECTIONS} ${BIBFILES} ${FIGS}

	${TEX} ${MAIN}; \
	biber ${MAIN}; \
	${TEX} ${MAIN}; \
	${TEX} ${MAIN}

#	@if ( grep "\\citation" ${MAIN}.aux > /dev/null); \
#	then \
#		echo '** Running BibTeX and then LaTeX again **'; \
#		bibtex ${MAIN}; \
#		${TEX} ${MAIN}; \
#	fi
	@while ( grep "Rerun to get " ${MAIN}.log > /dev/null ); \
	do \
		echo '** Re-running LaTeX **'; \
		${TEX} ${MAIN}; \
	done

	rm -f *.aux *.log *.blg *.bbl *.toc *.4?? *.css *.lg *.idv *.tmp *.xref *.out *.synctex.gz *.bcf *.xml

clean:
	rm -f *.aux *.log *.blg *.bbl *.toc *.4?? *.css *.lg *.idv *.tmp *.xref *.out *.synctex.gz *.bcf *.xml
# ${MAIN}.pdf