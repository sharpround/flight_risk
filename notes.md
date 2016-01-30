it looks like there would have to be a couple tiers, depending on how fine-grained the weather modeling is

1) predictions based on historical data
2) predictions based on macrotrends
3) predictions based on current weather

Those predictions need to be tied to each individual airport or flight.


To price a RG (rebook guarantee), what do you need?
* extreme weather values (probabilities) at all airports 
  - either by finding all the extreme weather in the country/world and then finding the impact on an airport/route, or by going through each airport/route and checking the weather...
  - what qualifies as "extreme weather"--what causes minor delays, what causes major delays (>4 hr), what causes cancellations (any different than just a major delay, from a Freedbird POV?), and what causes systemic cancellations (would make rebooking *much* more difficult and expensive?)
* what is the cost of rebooking

For *every* flight, we would need an estimate of the likelihood of a major delay or cancellation, and the estimated cost of automatic rebooking.


Can we think of the airline flight delays as a market? And each carrier performs relative to the market? Then we would have systemic behaviors and flight- or airline-dependent behaviors?


We want to do probabilistic classification (discrete choice, binomial regression). Options: naive bayes, logistic regression, neural net. (Others possibly, e.g. SVM). Start w/ logistic, because it is the easiest to interpret. Then random forest, just to see what we get out.


The good standard of a model  would be to backtest it on previous customers to see which made the best predictions.

Could we use a decision tree or a random forest as a clustering algorithm? (actually, I am not sure that we could use a random forest)

Also, need to keep in mind that even smaller delays matter, because they could cause a missed connection.

questions:
Why are flights delayed?
Are flights delayed more often in morning or afternoon?
Which carriers have delayed flights?
Which cities have delayed flights?
For a given city and time, are some carriers worse than others?
Do we have enough data to predict a delayed flight? (Nor even close, I assume.)
What is "dollar credibility indicator"?
How to get the rebooking price from the origin/destination survey data?
How do we add weather into that?
Can we use this data to identify "low-performing" planes?


todo:
stuff the data into a database



## Features, X:
Many of the features here are redundant. They should be condensed before modeling.

#### Time Period
Year  Year   
Quarter Quarter (1-4) Analysis
Month Month Analysis
DayofMonth  Day of Month   
DayOfWeek Day of Week Analysis
**FlightDate**  Flight Date (yyyymmdd)

#### Airline
**UniqueCarrier** Unique Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years. Analysis
AirlineID An identification number assigned by US DOT to identify a unique airline (carrier). A unique airline (carrier) is defined as one holding and reporting under the same DOT certificate regardless of its Code, Name, or holding company/corporation. Analysis
Carrier Code assigned by IATA and commonly used to identify a carrier. As the same code may have been assigned to different carriers over time, the code is not always unique. For analysis, use the Unique Carrier Code.  
*TailNum* Tail Number  
FlightNum Flight Number  

#### Origin
**OriginAirportID** Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused. Analysis
**OriginAirportSeqID**  Origin Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time.   
**OriginCityMarketID**  Origin Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market. Analysis
Origin  Origin Airport  Analysis
OriginCityName  Origin Airport, City Name  
**OriginState** Origin Airport, State Code  Analysis
OriginStateFips Origin Airport, State Fips  Analysis
OriginStateName Origin Airport, State Name   
OriginWac Origin Airport, World Area Code Analysis

#### Destination
**DestAirportID** Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.  Analysis
**DestAirportSeqID**  Destination Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time.  
**DestCityMarketID**  Destination Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market.  Analysis
Dest  Destination Airport Analysis
DestCityName  Destination Airport, City Name   
**DestState** Destination Airport, State Code Analysis
DestStateFips Destination Airport, State Fips Analysis
DestStateName Destination Airport, State Name  
DestWac Destination Airport, World Area Code  Analysis

#### Departure Performance
**CRSDepTime**  CRS Departure Time (local time: hhmm)  
**DepTimeBlk**  CRS Departure Time Block, Hourly Intervals  Analysis

#### Arrival Performance
**CRSArrTime**  CRS Arrival Time (local time: hhmm)  
**ArrTimeBlk**  CRS Arrival Time Block, Hourly Intervals  Analysis

#### Flight Summaries
**CRSElapsedTime**  CRS Elapsed Time of Flight, in Minutes  Analysis
Flights Number of Flights Analysis
**Distance**  Distance between airports (miles) Analysis
**DistanceGroup** Distance Intervals, every 250 Miles, for Flight Segment Analysis

## Targets, y
#### Cancellations and Diversions
Cancelled Cancelled Flight Indicator (1=Yes)  Analysis
CancellationCode  Specifies The Reason For Cancellation Analysis
Diverted  Diverted Flight Indicator (1=Yes) Analysis

#### Cause of Delay (Data starts 6/2003)
CarrierDelay  Carrier Delay, in Minutes Analysis
WeatherDelay  Weather Delay, in Minutes Analysis
NASDelay  National Air System Delay, in Minutes Analysis
SecurityDelay Security Delay, in Minutes  Analysis
LateAircraftDelay Late Aircraft Delay, in Minutes Analysis

#### Departure Performance
DepDelay  Difference in minutes between scheduled and actual departure time. Early departures show negative numbers.  Analysis
DepDelayMinutes Difference in minutes between scheduled and actual departure time. Early departures set to 0. Analysis
DepDel15  Departure Delay Indicator, 15 Minutes or More (1=Yes) Analysis
DepartureDelayGroups  Departure Delay intervals, every (15 minutes from <-15 to >180) Analysis
DepTimeBlk  CRS Departure Time Block, Hourly Intervals  Analysis
TaxiOut Taxi Out Time, in Minutes Analysis