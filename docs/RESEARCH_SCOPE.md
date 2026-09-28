# Research Scope

## Motivation

Conventional ML regression testing focuses on predictive performance and system behaviour. A model update can retain similar predictions while relying on different features. In safety-sensitive or governed settings, that change may require investigation even when accuracy remains stable.

## Core research question

How can explanation artifacts from two model versions be compared reliably, and when should an observed difference trigger an explanation-regression warning?

## Supporting questions

1. How much variation does the explanation method produce when the model does not change?
2. Which comparison metrics suit feature attributions, rankings and heatmaps?
3. How should local explanation changes be aggregated into a model-level warning?
4. How should thresholds be calibrated for a particular method and application?
5. Which versioned inputs are required to reproduce a comparison?

## CA1 proposal

The initial implementation will use tabular classification, SHAP and LIME, two model versions and a fixed probe set. It will estimate same-model variation before performing cross-version comparisons.

## Out of scope for the first implementation

- claiming that explanation drift proves causal or concept drift;
- automatic retraining decisions;
- image-based Grad-CAM comparison;
- production deployment and user-interface development;
- universal thresholds that apply to every model or dataset.

