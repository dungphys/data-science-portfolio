# Business Analysis Report — Automotive Resale Market

_Generated: 2026-08-05 00:03_

This report answers the business questions defined in the project 
README using the cleaned dataset and the trained resale-price model. 
See README.md for full methodology notes.

## Q1. Which vehicle attributes most influence resale price?

```
                   feature  importance
0         num__Vehicle_Age    0.455685
1             num__Mileage    0.085691
2  cat__Make_Mercedes-Benz    0.056161
3            cat__Make_BMW    0.054359
4           cat__Make_Audi    0.052250
5     cat__Body_Type_Sedan    0.034802
6       cat__Body_Type_SUV    0.028276
7              num__Owners    0.024180
8    num__Mileage_Per_Year    0.019984
9        cat__Make_Hyundai    0.013441
```

## Q2. How much does a prior accident reduce resale value?

```
                          mean   median  count  penalty_vs_no_accident_pct
Accident_History                                                          
No Accident       16337.673874  14119.0   2775                        0.00
Accident           6300.550967   3481.0   1913                       61.44
```

## Q3. What does the depreciation curve look like as vehicles age?

```
              median          mean  count  pct_of_newest_bucket
Vehicle_Age                                                    
2-3 yr       28839.5  32944.952465    568                 100.0
4-5 yr       23591.0  26387.270463    562                  81.8
6-8 yr       16343.0  18479.217722    790                  56.7
9-12 yr       9626.5  10997.260589   1086                  33.4
13+ yr         754.5   3077.513633   2494                   2.6
```

## Q4. Which fuel type / body type combinations command the highest resale value?

```
Fuel_Type   Diesel  Electric   Hybrid  Petrol
Body_Type                                    
Coupe       7970.0   13300.0   7329.0  5696.0
Hatchback    500.0    9380.0  12906.0  2507.0
SUV        12696.0   23326.0  17527.0  8164.0
Sedan       7890.0   12222.0  11798.0  6328.0
Truck      10267.0   18248.0  14965.0  9407.0
```

## Q5. Does a fuller service history translate into measurable resale value?

```
                         mean  median  count  premium_vs_no_service_pct
Service_History                                                        
No Service       11691.255507  8525.5    908                       9.75
Partial Service  12125.189247  8625.5   1860                       9.75
Full Service     12979.863612  9357.0   1877                       9.75
```

## Q6. Which states show the highest average resale prices?

```
                  mean  median  count
Location                             
GA        13142.795019  9691.5    522
OH        13116.835443  9052.5    474
CA        12774.747475  8661.0    495
TX        12647.325052  8525.0    483
MI        12596.272311  9424.0    437
IL        12300.588496  7979.5    452
NY        12206.705761  9000.0    486
FL        11865.568129  8632.0    433
NC        11642.582022  7567.0    445
PA        11579.519348  8023.0    491
```

## Q7. Does annualized mileage correlate with lower resale price?

```
correlation_mileage_per_year_vs_price   -0.5041
```
