# Avoiding Proxy Discrimination using Causal Reasoning

We try to solve discrimination on the <a href=https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29>German Credit Data Set</a> used in <a href=https://ieeexplore.ieee.org/abstract/document/8452913>Verma & Rubin (2018)</a> with the model presented in <a href=http://papers.nips.cc/paper/6668-avoiding-discrimination-through-causal-reasoning>Kilbertus et al. (2017)</a>.

The first thing we try is analysing the data and try to find a useful causal model to apply the method.
We build a Classifier implementing the idea from <a href=http://papers.nips.cc/paper/6668-avoiding-discrimination-through-causal-reasoning>Kilbertus et al. (2017)</a>. Since it is very difficult to set up a causal model, we restrict our model to the following features:

 - credit duration
 - credit history
 - credit amount
 - present employement since (proxy variable)
 - sex
 - age (protected attribute)
 - job
 - foreign worker
 
While it is for sure not a complete model it should be able to show if the idea the paper presents is working.

The first thing we have to do is to build a causal model graph.

The second thing we have to do is to reduce the dataset down to the needed features.

Then we can start applying the proposal from <a href=http://papers.nips.cc/paper/6668-avoiding-discrimination-through-causal-reasoning>Kilbertus et al. (2017)</a>.
