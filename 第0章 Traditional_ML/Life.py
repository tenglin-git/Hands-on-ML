
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

oecd_bli = pd.read_csv('oecd_bli_2017.csv', thousands=',')

gdp_per_capita = pd.read_csv('gdp_per_capita.csv', thousands=',')


def prepare_country_data(oecd_bli,gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli['Inequality'] == 'Total']
    oecd_bli = oecd_bli.pivot(index='Country', columns="Indicator", values='Value')

    gdp_per_capita = gdp_per_capita[gdp_per_capita['WEO Subject Code']=='NGDPDPC']

    gdp_per_capita.rename(columns={'2017':'GDP per capita'},inplace=True)
    gdp_per_capita.set_index('Country',inplace=True)

    full_country_data = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    full_country_data.sort_values(by='GDP per capita',inplace=True)

    return full_country_data[['GDP per capita','Life satisfaction']]


country_data = prepare_country_data(oecd_bli, gdp_per_capita)
#
# X = np.c_[country_data['GDP per capita']]
# #print(X)
# Y = np.c_[country_data['Life satisfaction']]

# country_data.plot(kind='scatter',x='GDP per capita',y='Life satisfaction')
# plt.show()



# model = LinearRegression()
# model.fit(X,Y)





