# Time-Series-Forecasting-of-Demand
Demand Forecasting through SARIMA and Winter Holt's Methods

GENERAL INFORMATION:

This project was undertaken as part of the RWTH Aachen Business School Analytics Project for Barkawi Group, a consultancy firm in the field of Supply Chain Optimization. The project was on Supply Chain analytics to evaluate the effect of re-balancing of inventory between retail outlets by modelling the uncertainties of demand using a stochastic program to optimize the supply plan.

The demand data was supplied by the consultants and as the first phase of the task, a demand forecasting was carried out during which the Winter Holt's Model of exponential smoothening was compared with the SARIMA method.

The code is attached here. This can be modified and used for any data. The data used for forecasting has not been shared.

A grid search was carried out in order to find the best parameters of the models. The models were then used to make forecasts of demand over the last six weeks, which also served as the test set and the RMS values of both model were compared to select the best model for prediction.

The best model was then trained over the whole data set and the results were used as input into the deterministic and stochastic dynamic programs deployed to optimize the supply plan.
