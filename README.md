# Review-CausalML: A Python Package for Causal Machine Learning

## Introduction

Causal Machine Learning (CausalML) is a Python library that implements causal inference techniques. In the past few years, people have become more interested in algorithms that combine causal inference and machine learning.
This package gives you access to a number of Python-written methods. The goal is to bridge the gap between theoretical work on methodologies and real-world applications. This review provides an overview of the package’s key concepts and its application domains.

The CausalML Python package offers state-of-the-art uplift modeling and [causal inference](https://en.wikipedia.org/wiki/Causal_inference) techniques using advanced machine learning algorithms. [The average treatment effect](https://en.wikipedia.org/wiki/Average_treatment_effect) (ATE) of a treatment or intervention can be estimated using conventional causal analysis techniques like a t-test on randomized trials (also known as A/B testing).

But in many applications, it’s useful to have a finer-grained approximation of these effects. To estimate the effect at the individual or segment level, CausalML provides the user with the possibility to estimate the [conditional average treatment effect](https://arxiv.org/abs/2104.09935) (CATE). By giving each customer a treatment that fits their needs based on these predictions, a lot of optimization and personalization options open up.

# What is uplift modeling?

Uplift modeling is a powerfull modeling technique made possible by CausalML. Uplift modeling is a set of methods a business can use to predict the positive or negative impact of an action on a particular customer outcome.
Customer relationship management, promotions, incentives, advertisements, customer service, recommendation systems, and even product design all make use of it to better target their customers and allocate their budgets.

An optimal treatment strategy is provided after assessing the ITE or CATE of the treatment for a user or set of users, taking into account the potential lift by and cost of the treatment. After getting a promotional email, a manager at a telecommunications company can predict how many customers who fit a certain profile will renew their service in the next billing cycle.

“Uplift modeling enables us to design a refined decision rule for optimally determining whether to treat each individual or not, depending on his or her features. Such a treatment rule allows us to only target those who positively respond to the treatment and avoid treating negative responders.”

### Uplift modeling process

 - Researchers will submit a random sample of the population to the action being analyzed (treatment dataset).
 - Another disjointed, random sample is also selected, to which the action is not applied. This is the control dataset, which will be used as a baseline to see how well the action worked.
 - Now that we have two sets of data to work with (treatment and control), we can create a model that predicts the difference between the two sets of data rather than the probability of objects belonging to a specific class.
    
    ## CausalML: Python package for causal machine learning
    
    Conducting a randomized experiment to draw causal inferences is not something that this package is meant to replace. Evaluation of the average treatment effect (ATE) for business decisions sometimes requires randomized experiments. Even though uplift modeling can be used with experimental and observational data, this particular implementation is best used with data from a randomized experiment.

“Applications to observational data where the treatment is not assigned randomly should take extra caution. In a non-randomized experiment, there is often a selection bias in the treatment assignment (a.k.a. the confounding effect). One main challenge is that omitting potential confounding variables from the model can produce biased estimation for the treatment effect. On the other hand, properly randomized experiments
do not suffer from such selection bias, that provides a better basis for uplift modeling to estimate the CATE (or individual level lift).”

### Python packages for causalML

There are a few packages related to CausalML:

 - Uplift, grf, and rlearn packages are used to implement the [uplift random forest](https://link.springer.com/article/10.1007/s10618-014-0383-9), [generalized random forest](https://grf-labs.github.io/grf/), and R-learner techniques.
 - [DoWhy](https://github.com/py-why/dowhy) Python module uses graphical models inspired by Judea Pearl’s do-calculus and the potential outcomes framework to provide a structured approach to the causal inference problem.
 - [EconML](https://github.com/microsoft/EconML) Python module was made so that machine learning techniques could be used with heterogeneous treatment effect estimators from econometrics(such as instrumental variables).
 - [Pylift](https://pylift.readthedocs.io/en/latest/) implements one metalearner for uplift modeling. The current version of the CausalML package contributes by acting as a central hub for uplift modeling techniques.

## Why CausalML ?

Causal inference and machine learning have been a popular academic topics. The experience of researchers at Uber has led us to think that this study will produce real-world applications. Their goal with the CausalML package was to increase the number of people who could use these applications. The end objective of their work was to provide a unified environment for causal inference using machine learning.

The purpose of the first release of the CausalML package was to make uplift modeling techniques more accessible to a large audience. To achieve this goal, they provide a uniform application programming interface (API) that enables running an uplift method as simple as fitting a standard classification or regression model. Several metrics and visualization tools, such as uplift curves, are provided for assessing the model’s performance.

## Algorithms supported by CausalML

The package currently supports many algorithms, but we will mention some of them:

**Tree-based algorithms**:

- [Uplift trees and random forests](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf) on KL divergence, Euclidean distance, and Chi-Square.
- [Uplift trees or random forests](https://arxiv.org/abs/1705.08492) based on contextual treatment selection.
- [Causal Tree](https://github.com/susanathey/causalTree?utm_source=catalyzex.com) — WWork in Progress

**Meta-learner algorithms**:

- S-learner: With only one machine learning model, S-learner can estimate the treatment effect.
- T-Learner: [T-Learner](https://arxiv.org/abs/1706.03461) is a two-step process. In the first step, the control response function is estimated using data from the control group by a base learner, which can be any supervised learning or regression estimator. Second, the treatment response function is estimated.
- [X-Learner](https://arxiv.org/abs/1706.03461): X-Learner can be described in three stages: First, estimate the response functions using any supervised learning or regression algorithm and denote the estimated functions. Second, impute the user-level treatment effects. Third, define the CATE estimate by a weighted average.
- R-learner: [R-learner](https://github.com/xnie/rlearner?utm_source=catalyzex.com) uses the out-of-fold estimates of outcomes and propensity scores from cross-validation.
- Doubly robust (DR) learner: [DR-learner](https://www.math.mcgill.ca/dstephens/SISCR2018/Articles/bang_robins_2005.pdf) estimates the CATE by cross-fitting a doubly robust score function in two stages.
- TMLE learner: The [Targeted Maximum Likelihood Estimation](https://www.khstats.com/blog/tmle/tutorial) (TMLE) framework is used to estimate a statistical quantity of interest.
- TMLE supports the application of machine learning (ML) models with little assumptions about data distribution. Unlike ML estimates, the final TMLE estimate will still include appropriate standard errors for statistical inference.

You can look at the [documentation](https://causalml.readthedocs.io/en/latest/) for more information. Researchers designed the package to be versatile in terms of the types of outcome variables that can be modeled, supporting both regression and classification tasks. The package also includes algorithms for analyzing data from experiments with multiple treatment groups.

## What problems can CausalML solve?

CausalMLs use cases include, but are not limited to, targeting optimization, engagement personalization, and causal impact analysis.

### Targeting Optimization

- We can use CausalML to target promotions at people who will provide the most value.
- In a cross-sell marketing campaign for current customers, we can provide promotions to customers who are more likely to use a new product, especially - because of the exposure to the promotions, while conserving inbox space for others.
- Internal research has shown that using uplift modeling on only 30% of users could have the same effect on sales of a new product as offering the promotion to all customers.

### Causal impact analysis

- The comprehensive capabilities of CausalML allow us to analyze the effects of a specific event on experimental or observational data by incorporating rich features.
- We can study the effect of a cross-sell on a customer’s future platform expenditures. Setting up a randomized test would be impractical because we don’t want to prevent any clients from making the switch to the new product.
- With the help of CausalML, we can run some ML-based causal inference algorithms to figure out how the cross-sell affects the whole platform.

### Personalization

- To tailor the user experience, you can use CausalML. A business can communicate with its clients in many ways, such as through different product choices for upselling or messaging channels for communications.
- Using CausalML, one can determine the ideal tailored offer for each client by estimating the effect of each possible combination.

## Conclusion

The Uber CausalML team is always working to improve and update the package. The team want to improve the computational efficiency of current algorithms in the package. They plan to include more cutting-edge uplift models. In addition to uplift modeling, they are investigating other modeling approaches at the confluence of machine learning and causal inference to resolve optimization problems.


# Installation

Installation with `conda` is recommended. `conda` environment files for Python 3.6, 3.7, 3.8 and 3.9 are available in the repository. To use models under the `inference.tf` module (e.g. `DragonNet`), additional dependency of `tensorflow` is required. For detailed instructions, see below.

## Install using `conda`:
### Install from `conda-forge`
Directly install from the conda-forge channel using conda.

```sh
$ conda install -c conda-forge causalml
```

### Install with the `conda` virtual environment
This will create a new `conda` virtual environment named `causalml-[tf-]py3x`, where `x` is in `[6, 7, 8, 9]`. e.g. `causalml-py37` or `causalml-tf-py38`. If you want to change the name of the environment, update the relevant YAML file in `envs/`

```
$ git clone https://github.com/uber/causalml.git
$ cd causalml/envs/
$ conda env create -f environment-py38.yml	# for the virtual environment with Python 3.8 and CausalML
$ conda activate causalml-py38
(causalml-py38)
```

### Install `causalml` with `tensorflow`
```
$ git clone https://github.com/uber/causalml.git
$ cd causalml/envs/
$ conda env create -f environment-tf-py38.yml	# for the virtual environment with Python 3.8 and CausalML
$ conda activate causalml-tf-py38
(causalml-tf-py38) pip install -U numpy			# this step is necessary to fix [#338](https://github.com/uber/causalml/issues/338)
```

## Install using `pip`:

```
$ git clone https://github.com/uber/causalml.git
$ cd causalml
$ pip install -r requirements.txt
$ pip install causalml
```

### Install `causalml` with `tensorflow`
```
$ git clone https://github.com/uber/causalml.git
$ cd causalml
$ pip install -r requirements-tf.txt
$ pip install causalml[tf]
$ pip install -U numpy							# this step is necessary to fix [#338](https://github.com/uber/causalml/issues/338)
```

## Install from source:

```
$ git clone https://github.com/uber/causalml.git
$ cd causalml
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace
$ python setup.py install
```


# Quick Start

## Average Treatment Effect Estimation with S, T, X, and R Learners

```python
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.inference.meta import BaseRRegressor
from xgboost import XGBRegressor
from causalml.dataset import synthetic_data

y, X, treatment, _, _, e = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)

lr = LRSRegressor()
te, lb, ub = lr.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Linear Regression): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

xg = XGBTRegressor(random_state=42)
te, lb, ub = xg.estimate_ate(X, treatment, y)
print('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

nn = MLPTRegressor(hidden_layer_sizes=(10, 10),
                 learning_rate_init=.1,
                 early_stopping=True,
                 random_state=42)
te, lb, ub = nn.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Neural Network (MLP)): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

xl = BaseXRegressor(learner=XGBRegressor(random_state=42))
te, lb, ub = xl.estimate_ate(X, treatment, y, e)
print('Average Treatment Effect (BaseXRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

rl = BaseRRegressor(learner=XGBRegressor(random_state=42))
te, lb, ub =  rl.estimate_ate(X=X, p=e, treatment=treatment, y=y)
print('Average Treatment Effect (BaseRRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))
```

See the [Meta-learner example notebook](https://github.com/uber/causalml/blob/master/examples/meta_learners_with_synthetic_data.ipynb) for details.


## Interpretable Causal ML

Causal ML provides methods to interpret the treatment effect models trained as follows:

### Meta Learner Feature Importances

```python
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.dataset.regression import synthetic_data

# Load synthetic data
y, X, treatment, tau, b, e = synthetic_data(mode=1, n=10000, p=25, sigma=0.5)
w_multi = np.array(['treatment_A' if x==1 else 'control' for x in treatment]) # customize treatment/control names

slearner = BaseSRegressor(LGBMRegressor(), control_name='control')
slearner.estimate_ate(X, w_multi, y)
slearner_tau = slearner.fit_predict(X, w_multi, y)

model_tau_feature = RandomForestRegressor()  # specify model for model_tau_feature

slearner.get_importance(X=X, tau=slearner_tau, model_tau_feature=model_tau_feature,
                        normalize=True, method='auto', features=feature_names)

# Using the feature_importances_ method in the base learner (LGBMRegressor() in this example)
slearner.plot_importance(X=X, tau=slearner_tau, normalize=True, method='auto')

# Using eli5's PermutationImportance
slearner.plot_importance(X=X, tau=slearner_tau, normalize=True, method='permutation')

# Using SHAP
shap_slearner = slearner.get_shap_values(X=X, tau=slearner_tau)

# Plot shap values without specifying shap_dict
slearner.plot_shap_values(X=X, tau=slearner_tau)

# Plot shap values WITH specifying shap_dict
slearner.plot_shap_values(X=X, shap_dict=shap_slearner)

# interaction_idx set to 'auto' (searches for feature with greatest approximate interaction)
slearner.plot_shap_dependence(treatment_group='treatment_A',
                              feature_idx=1,
                              X=X,
                              tau=slearner_tau,
                              interaction_idx='auto')
```
<div align="center">
  <img width="629px" height="618px" src="https://raw.githubusercontent.com/uber/causalml/master/docs/_static/img/shap_vis.png">
</div>

See the [feature interpretations example notebook](https://github.com/uber/causalml/blob/master/examples/feature_interpretations_example.ipynb) for details.

### Uplift Tree Visualization

```python
from IPython.display import Image
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot

uplift_model = UpliftTreeClassifier(max_depth=5, min_samples_leaf=200, min_samples_treatment=50,
                                    n_reg=100, evaluationFunction='KL', control_name='control')

uplift_model.fit(df[features].values,
                 treatment=df['treatment_group_key'].values,
                 y=df['conversion'].values)

graph = uplift_tree_plot(uplift_model.fitted_uplift_tree, features)
Image(graph.create_png())
```
<div align="center">
  <img width="800px" height="479px" src="https://raw.githubusercontent.com/uber/causalml/master/docs/_static/img/uplift_tree_vis.png">
</div>

See the [Uplift Tree visualization example notebook](https://github.com/uber/causalml/blob/master/examples/uplift_tree_visualization.ipynb) for details.

# Contributing

We welcome community contributors to the project. Before you start, please read our [code of conduct](https://github.com/uber/causalml/blob/master/CODE_OF_CONDUCT.md) and check out [contributing guidelines](./CONTRIBUTING.md) first.


# Versioning

We document versions and changes in our [changelog](https://github.com/uber/causalml/blob/master/docs/changelog.rst).


# License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/uber/causalml/blob/master/LICENSE) file for details.


# References

## Documentation
* [Causal ML API documentation](https://causalml.readthedocs.io/en/latest/about.html)

## Conference Talks and Publications by CausalML Team
* (Talk) Introduction to CausalML at [Causal Data Science Meeting 2021](https://www.causalscience.org/meeting/program/day-2/)
* (Talk) Introduction to CausalML at [2021 Conference on Digital Experimentation @ MIT (CODE@MIT)](https://ide.mit.edu/events/2021-conference-on-digital-experimentation-mit-codemit/)
* (Talk) Causal Inference and Machine Learning in Practice with EconML and CausalML: Industrial Use Cases at Microsoft, TripAdvisor, Uber at [KDD 2021 Tutorials](https://kdd.org/kdd2021/tutorials) ([website and slide links](https://causal-machine-learning.github.io/kdd2021-tutorial/))
* (Publication) CausalML White Paper [Causalml: Python package for causal machine learning](https://arxiv.org/abs/2002.11631)
* (Publication) [Uplift Modeling for Multiple Treatments with Cost Optimization](https://ieeexplore.ieee.org/document/8964199) at [2019 IEEE International Conference on Data Science and Advanced Analytics (DSAA)](http://203.170.84.89/~idawis33/dsaa2019/preliminary-program/)
* (Publication) [Feature Selection Methods for Uplift Modeling](https://arxiv.org/abs/2005.03447)

## Citation
To cite CausalML in publications, you can refer to the following sources:

Whitepaper:
[CausalML: Python Package for Causal Machine Learning](https://arxiv.org/abs/2002.11631)

Bibtex:
> @misc{chen2020causalml,
>    title={CausalML: Python Package for Causal Machine Learning},
>    author={Huigang Chen and Totte Harinen and Jeong-Yoon Lee and Mike Yung and Zhenyu Zhao},
>    year={2020},
>    eprint={2002.11631},
>    archivePrefix={arXiv},
>    primaryClass={cs.CY}
>}


## Literature

1. Chen, Huigang, Totte Harinen, Jeong-Yoon Lee, Mike Yung, and Zhenyu Zhao. "Causalml: Python package for causal machine learning." arXiv preprint arXiv:2002.11631 (2020).
2. Radcliffe, Nicholas J., and Patrick D. Surry. "Real-world uplift modelling with significance-based uplift trees." White Paper TR-2011-1, Stochastic Solutions (2011): 1-33.
3. Zhao, Yan, Xiao Fang, and David Simchi-Levi. "Uplift modeling with multiple treatments and general response types." Proceedings of the 2017 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2017.
4. Athey, Susan, and Guido Imbens. "Recursive partitioning for heterogeneous causal effects." Proceedings of the National Academy of Sciences 113.27 (2016): 7353-7360.
5. Künzel, Sören R., et al. "Metalearners for estimating heterogeneous treatment effects using machine learning." Proceedings of the national academy of sciences 116.10 (2019): 4156-4165.
6. Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment effects." arXiv preprint arXiv:1712.04912 (2017).
7. Bang, Heejung, and James M. Robins. "Doubly robust estimation in missing data and causal inference models." Biometrics 61.4 (2005): 962-973.
8. Van Der Laan, Mark J., and Daniel Rubin. "Targeted maximum likelihood learning." The international journal of biostatistics 2.1 (2006).
9. Kennedy, Edward H. "Optimal doubly robust estimation of heterogeneous causal effects." arXiv preprint arXiv:2004.14497 (2020).
10. Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." arXiv preprint arXiv:1705.08821 (2017).
11. Shi, Claudia, David M. Blei, and Victor Veitch. "Adapting neural networks for the estimation of treatment effects." 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), 2019.
12. Zhao, Zhenyu, Yumin Zhang, Totte Harinen, and Mike Yung. "Feature Selection Methods for Uplift Modeling." arXiv preprint arXiv:2005.03447 (2020).
13. Zhao, Zhenyu, and Totte Harinen. "Uplift modeling for multiple treatments with cost optimization." In 2019 IEEE International Conference on Data Science and Advanced Analytics (DSAA), pp. 422-431. IEEE, 2019.
 

## Related projects

* [uplift](https://cran.r-project.org/web/packages/uplift/index.html): uplift models in R
* [grf](https://cran.r-project.org/web/packages/grf/index.html): generalized random forests that include heterogeneous treatment effect estimation in R
* [rlearner](https://github.com/xnie/rlearner): A R package that implements R-Learner
* [DoWhy](https://github.com/Microsoft/dowhy):  Causal inference in Python based on Judea Pearl's do-calculus
* [EconML](https://github.com/microsoft/EconML): A Python package that implements heterogeneous treatment effect estimators from econometrics and machine learning methods
