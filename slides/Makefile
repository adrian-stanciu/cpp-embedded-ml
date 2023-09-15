TARGET := slides
OUTDIR := texfiles

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex sections/*.tex
	test -d $(OUTDIR) || mkdir $(OUTDIR)
	pdflatex -output-directory $(OUTDIR) $<
	pdflatex -output-directory $(OUTDIR) $< # Twice, so ToC is also updated
	cp $(OUTDIR)/$(TARGET).pdf .

.PHONY: clean
clean:
	rm -rf $(OUTDIR)
	rm -f $(TARGET).pdf

