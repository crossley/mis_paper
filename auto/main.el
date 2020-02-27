(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("apa6" "jou" "11pt" "longtable" "floatsintext" "notab")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("biblatex" "style=apa" "backend=biber" "citestyle=numeric" "sorting=none")))
   (TeX-run-style-hooks
    "latex2e"
    "apa6"
    "apa611"
    "epsf"
    "epsfig"
    "amssymb"
    "amsmath"
    "xcolor"
    "booktabs"
    "colortbl"
    "rotating"
    "setspace"
    "fancyhdr"
    "float"
    "inputenc"
    "fontenc"
    "textcomp"
    "gensymb"
    "biblatex")
   (LaTeX-add-labels
    "fig_paradigm"
    "eq_simple_err"
    "eq_simple_x"
    "eq_context_x_2"
    "eq_context_err"
    "eq_dual_rate_dual_1"
    "eq_dual_rate_dual_2"
    "eq_dual_rate_dual_3"
    "eq_dual_rate_dual_4"
    "eq_sse"
    "eq_sse_boot"
    "fig_results"
    "table_1"
    "fig_params")
   (LaTeX-add-bibliographies
    "mis_refs"))
 :latex)

