.PHONY: all clean pdf bib
.SUFFIXES:

FILENAME := main
LATEX := -pdflatex
LATEXFLAGS := -synctex=1 -shell-escape -interaction=nonstopmode
BIB := -bibtex
BIBFLAGS :=

all:
	$(LATEX) $(LATEXFLAGS) $(FILENAME).tex
	$(BIB) $(BIBFLAGS) $(FILENAME)
	$(LATEX) $(LATEXFLAGS) $(FILENAME).tex
	rm -fv $(FILENAME).{bbl,bcf,blg,lof,run.xml,synctex.gz,toc,fdb_latexmk,fls,tdo,listing,auxlock,out.ps,aux,log,out}

auto:
	latexmk -f -pdf -pvc $(FILENAME)

pdf:
	$(LATEX) $(LATEXFLAGS) $(FILENAME).tex

bib:
	$(BIB) $(BIBFLAGS) $(FILENAME)

clean:
	rm -fv $(FILENAME).{aux,bbl,bcf,blg,lof,log,pdf,run.xml,synctex.gz,\
	ind,idx,ilg,toc,fdb_latexmk,fls,tdo,listing,auxlock,aux,log,out}
	rm -fv "*.synctex*"
	rm -fv *-blx.bib

gitpush:
	rm -fv $(FILENAME).{aux,bbl,bcf,blg,lof,log,pdf,run.xml,synctex.gz,\
	ind,idx,ilg,toc,fdb_latexmk,fls,tdo,listing,auxlock,aux,log,out}
	rm -fv *-blx.bib
	git add .
	git commit -a --allow-empty-message -m ''
	git push
#	#in case there are problems maybe force the push with "-f",\
#	#but be aweare of what you are really doing!
