# Laboratories – Simulation and Performance Evaluation @ UNITN

This repository contains the homework assignments for the **Laboratories of the Simulation and Performance Evaluation** course at the University of Trento (UNITN).

## Structure

Each homework folder contains:
- A Jupyter Notebook with the solution
- A PDF version of the notebook
- A LaTeX report (both normal and anonymous versions)

---

## 📝 Compiling the LaTeX Report

You can compile the report in both **normal** and **anonymous** mode.

### With VS Code

If you're using **LaTeX Workshop** on VS Code:
1. Open the `.tex` file
2. Press `Ctrl+Shift+P` and select  
   `LaTeX Workshop: Build with recipe → Compile both`  
   This will generate both the named and anonymous PDFs.

### From the Command Line

To compile the **anonymous** version manually:

```bash
cd <HOMEWORK_FOLDER>
pdflatex -interaction=nonstopmode -synctex=1 -jobname=anonymous_report "\def\anonymous{} \input{report.tex}"
```

To compile the **named** version:

```bash
pdflatex report.tex
```

---

## 📓 Converting Jupyter Notebooks to PDF

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. If you have [Pandoc](https://pandoc.org/installing.html) installed:

```bash
jupyter nbconvert --to pdf <NOTEBOOK_PATH>
```

3. If you don't have Pandoc:

```bash
jupyter nbconvert --to html <NOTEBOOK_PATH>
```

Then convert the HTML file to PDF using your preferred tool.

