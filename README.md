demand_forecasting_simulation
=============================

This is a package to create simulated data that can be used for developement of statistical models for Demand Forecasting, i.e., time series predictions for different products in different locations, for example of a grocery chain.

Usage
-----

Just run simulations.py, which includes the main function.

Created Data Sets
-----------------

There are three different data sets:

**train.csv**:
Meant to be used for the training of the statistical models.

**test.csv**:
Can be considered as future data and is meant to be predicted out-of-sample (i.e., aim is to achieve generalization in the training).

**test_results.csv**:
Contains the actual values to be predicted from test.csv. Only meant for model evaluation and must not be used in the training or prediction processes.


Description of Variables
------------------------

**P_ID**
Identifier of the different products/articles.

**L_ID**
Identifier of the different locations/stores.

**DATE**
Date of the data of the given record. Together with P_ID and L_ID, uniquely identifies each record.

**SALES**
Aggregated daily sales of the given product in the given location at the given date.

**PG_ID_1**
Identifier of the highest level of the product group hierarchy, e.g., food or beverages.

**PG_ID_2**
Identifier of the second highest level of the product group hierarchy, e.g., sweets or vegetables.

**PG_ID_3**
Identifier of the lowest level of the product group hierarchy, e.g., apples or bananas.

**NORMAL_PRICE**
Usual, unreduced price of the product-location-date combination at hand. Not necessarily the actual sales price (in case of a price reduction).

**SALES_PRICE**
Actual, potentially reduced, price of the product-location-date combination at hand.

**PROMOTION_TYPE**
Values indicating no promotion (0) or two different typess of promotions (1 and 2) for the product-location-date combination at hand.

**SALES_AREA**
Size of the given store in square meters.

**EVENT**
Name of a holiday occuring on the given date in the region of the given location.

**SCHOOL_HOLIDAY**
Boolean flag indicating if the given date is a school holiday in the region of the given location.
