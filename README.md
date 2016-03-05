
# auto-clean
[![Travis-CI Build Status](https://travis-ci.org/ericfourrier/auto-clean.svg?branch=develop)](https://travis-ci.org/ericfourrier/auto-clean)  

Provide helpers in python (pandas, numpy, scipy) to perform automated cleaning and pre-processing steps.


## CONTENTS OF THIS FILE


 * Introduction
 * Installation
 * Version
 * Usage
 * Thanks


## INTRODUCTION


This package provide different classes for data cleaning

In this module you have :

 * An `DataExploration` class with methods to count missing values, detect constant col
the `DataExploration.structure` provide a nice exploration summary.

 * An `OutliersDetection` class which is a simple class designed to detect 1d outliers

 * An `NaImputer` class which is a simple class designed to focus on missing values
 exploration and comprehension (Rubbin theory MCAR/MAR/MNAR)

 * A examples folder to find notebooks illustrating the package

 * A `test.py`  file containing tests (you can run the test with `$python -m unittest -v test`)

## INSTALLATION

Installation via pip is not available now (*coming soon*)

 1. Clone the project on your local computer.

 2. Run the following command

 	* `$ python setup.py install`

## VERSION

The current version is 0.1 (early release version).
The module will be improved over time.

## USAGE
To complete

## Contributing to auto-clean
Anybody is welcome  to do pull-request and check the existing [issues](https://github.com/ericfourrier/auto-clean/issues) for bugs or enhancements to work on.
If you have an idea for an extension to auto-clean, create a new issue so we can discuss it.

## THANKS
Thanks to all the creator and contributors of the package we are using.
