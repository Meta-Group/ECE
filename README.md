# ECE: Ensemble of Counterfactual Explainers
**Riccardo Guidotti, Salvatore Ruggieri**    
Department of Computer Science, University of Pisa, Italy  
riccardo.guidotti@unipi.it, salvatore.ruggieri@unipi.it

In eXplainable Artificial Intelligence (XAI), several counterfactual explainers have been proposed, each focusing on some desirable
properties of counterfactual instances: minimality, actionability, stability, diversity, plausibility, discriminative power. We propose an ensemble
of counterfactual explainers that boosts weak explainers, which provide only a subset of such properties, to a powerful method covering all of
them. The ensemble runs weak explainers on a sample of instances and of features, and it combines their results by exploiting a diversity-driven
selection function. The method is model-agnostic and, through a wrapping approach based on autoencoders, it is also data-agnostic


## References

[1] R. Guidotti, S. Ruggieri. [Ensemble of Counterfactual Explainers](http://pages.di.unipi.it/ruggieri/Papers/ds2021.pdf). Discovery Science (DS 2021). 358-368. Vol. 12986 of LNCS, Springer, October 2021.

## How to install required packages

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Adding Scamandar package directory to pythonpath so that we can import the experiments as is

```bash
cat >> .venv/bin/activate <<'BASH'
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASH
```
In case the venv is active, you have to run `deactivate` and `source .venv/bin/activate` again to take effect.
