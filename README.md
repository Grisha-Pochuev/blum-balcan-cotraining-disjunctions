# Co-training with disjunctions: an efficient semi-supervised learner

**Efficient learning from disjunction agreement pairs — an algorithm and proof for the formulation in Blum–Balcan (2007)**

Grisha Pochuev · ORCID [0009-0005-0078-0857](https://orcid.org/0009-0005-0078-0857) · <n_854@mail.ru> · research note dated 20 October 2026

📄 **Paper:** [`paper/disjunction_agreement_note.pdf`](paper/disjunction_agreement_note.pdf) (8 pages, LaTeX source alongside)

---

## The question

In *Open Problems in Efficient Semi-Supervised PAC Learning* (COLT 2007, [PDF](https://home.ttic.edu/~avrim/Papers/openproblems.pdf), [DOI 10.1007/978-3-540-72927-3_45](https://doi.org/10.1007/978-3-540-72927-3_45)), Avrim Blum and Maria-Florina Balcan ask for computationally efficient algorithms in their semi-supervised PAC model that learn with roughly

$$O\Big(\tfrac{1}{\varepsilon}\big(\log |C_{D,\chi}(\varepsilon)| + \log \tfrac{1}{\delta}\big)\Big)$$

labelled examples, where $C_{D,\chi}(\varepsilon)$ is the set of hypotheses that are nearly compatible with the distribution. One of the listed instances (Section 3.2) is

> **Co-training with disjunctions.** Let $C$ be the class of disjunctions, but an example $x$ is a pair of points $(x_1, x_2)$ in $\{0,1\}^n$. Define $h(x) = h(x_1)$ but say that $h$ is compatible with $x$ iff $h(x_1) = h(x_2)$.

## The result

For **any** distribution $D$ over pairs (the two views may be arbitrarily dependent and have different marginals) and any target disjunction $h^*$ that agrees on both views with probability 1, a polynomial-time algorithm returns a disjunction $\hat h$ with prediction error at most $\varepsilon$ on both views and incompatibility at most $\tau$, with probability at least $1-\delta$, using

| resource | amount |
|---|---|
| unlabelled pairs | $u \ge \dfrac{d\ln 2 + \ln(2/\delta)}{\tau}$ |
| labelled pairs | $m \ge \dfrac{\ln K + \ln(2/\delta)}{\varepsilon}$, where $K = \lvert C_{D,\chi}(\tau)\rvert$ |
| time | $O(d(u+m)+d)$ after feature evaluation |

with $d = n$ for monotone disjunctions and $d = 2n$ when negated literals are allowed. Taking $\tau = \varepsilon$ gives the labelled-sample benchmark of the 2007 question. No conditional independence of the views, no margin parameter, and no bound on the number of relevant variables is assumed.

Section 4 of the note adds a fully specified stopping rule (geometrically growing training blocks with independent validation) that does **not** need to know $K$, and uses $O((\ln K + \ln(1/\delta))/\varepsilon)$ labels both in expectation and with high probability.

## The idea in one paragraph

Write a disjunction as its support $S$ and let $E = [d]\setminus S$ be the excluded features. A point is classified negative iff all its active features lie in $E$. So for an unlabelled pair with active sets $A, B$,

$$h_S(x_1) = h_S(x_2) \iff \big(A \subseteq E \Leftrightarrow B \subseteq E\big),$$

i.e. two Horn implications: *if $A$ is fully excluded, exclude $B$*, and vice versa. Start by excluding every feature that appears in a negatively labelled pair, then propagate these implications to a fixed point. Target features are never excluded, so the remaining support is the **greatest** disjunction consistent with all labels and all unlabelled agreements; positive labels are satisfied automatically. Standard finite-class bounds then finish the proof: the unlabelled sample shrinks the surviving hypotheses to $C_{D,\chi}(\tau)$, and the labels only need to pick among those $K$ functions.

## Repository layout

```
paper/
  disjunction_agreement_note.pdf     research note (8 pages)
  disjunction_agreement_note.tex     LaTeX source
  LICENSE-CC-BY-4.0.txt              licence for the paper
code/
  cotrain_disjunctions.py            queue-based solver, literal lifting, PAC wrapper with stopping rule
  verify_solver.py                   implementation vs. complete enumeration
  independent_verify.py              independently written enumerator for the closure property
  wrapper_smoke_check.py             seeded runs of the stopping rule
  *_results.json                     exact outputs of the three checks
docs/
  README_ru.txt                      summary in Russian
  sources_and_status.txt             sources consulted and bibliographic status (Russian)
SHA256SUMS.txt                       checksums of all files
CITATION.cff, .zenodo.json           citation and archive metadata
```

## Reproducing

Python 3.10+ and the standard library only.

```bash
cd code
python cotrain_disjunctions.py --demo   # worked example from Section 6 of the note
python verify_solver.py                 # ~15 s, rewrites solver_verification_results.json
python independent_verify.py            # ~3 s, rewrites independent_verify_results.json
python wrapper_smoke_check.py           # ~1 s, rewrites wrapper_smoke_results.json
```

All runs are seeded; the regenerated JSON files are byte-identical to the committed ones.

| check | what it covers | result |
|---|---|---|
| `verify_solver.py` | 22,560 constraint systems vs. full enumeration, incl. 2,000 signed-literal and 1,000 arbitrary (possibly inconsistent) systems | passed: 160,627 feasible-support witnesses, 71,335 certificate steps replayed, 746 inconsistent systems correctly rejected |
| `independent_verify.py` | 19,560 systems: exhaustive two-pair systems for $d \le 3$, random systems for $4 \le d \le 8$, 4,000 target-generated labelled samples | passed, 48,000 labelled pairs checked |
| `wrapper_smoke_check.py` | stopping rule: cap, rejection/continuation, validation acceptance | passed |

These are finite consistency checks, not a machine-checked formalization; the probabilistic theorems are proved in the note.

## Related work

- A. Blum, T. Mitchell. *Combining labeled and unlabeled data with co-training.* COLT 1998 — co-training under independence of views given the label.
- M.-F. Balcan, A. Blum. *A Discriminative Model for Semi-Supervised Learning.* JACM 57(3), 2010 ([author version](https://home.ttic.edu/~avrim/Papers/labeled_unlabeled-2009.pdf)) — the framework and the $\log|C_{D,\chi}(\varepsilon)|$ bound.
- M.-F. Balcan, C. Berlind, S. Ehrlich, Y. Liang. *Efficient Semi-supervised and Active Learning of Disjunctions.* ICML 2013 ([PDF](https://proceedings.mlr.press/v28/balcan13.pdf)) — the different *two-sided disjunctions* item, where the consistency problem is NP-hard.
- A. Blum, Y. Mansour. *Efficient Co-Training of Linear Separators under Weak Dependence.* COLT 2017 ([PDF](http://proceedings.mlr.press/v65/blum17a/blum17a.pdf)).
- I. Diakonikolas, M. Ma, L. Ren, C. Tzamos. *Fast Co-Training under Weak Dependence via Stream-Based Active Learning.* ICML 2024 ([page](https://proceedings.mlr.press/v235/diakonikolas24b.html)).

## Status

This note supplies a self-contained mathematical argument for the stated learning guarantee. It does not establish that the argument is new, that the problem remained open until this date, or that the historical reward is currently payable; those are separate bibliographic and author-review questions. No external acceptance is claimed. The note was prepared with GPT-6 Astra assistance.

## Citation

See [`CITATION.cff`](CITATION.cff). A Zenodo DOI will be added here once the record is published.

## Licence

Paper (`paper/`, `docs/`): [CC BY 4.0](paper/LICENSE-CC-BY-4.0.txt). Code (`code/`): [MIT](LICENSE).

Contact: Grisha Pochuev, <n_854@mail.ru>
