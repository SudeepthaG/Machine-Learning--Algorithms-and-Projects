'''
    Author:
    Arpit Parwal <aparwal@usc.edu>
    Yeon-soo Park <yeonsoop@usc.edu>
    Vanessa Tan <tanvanes@usc.edu>
    Sudeeptha Mouni Ganji <sganji@usc.edu>

'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import plotly.subplots as tls
import plotly.graph_objs as go
import plotly
from sklearn.feature_selection import mutual_info_regression




INPUT_FILE = "./dataSource/features_combined.csv"

## Add feature names that you want to filter out
NULL_FEATURES = ['country', 'Country_Region', 'entity', 'total_covid_19_tests']

## Add Features that you want to load from features_combined.csv
FILTER_FEATURES = ['Country_Region', 'total_covid_19_tests', 'Confirmed', 'pop2020',
                   'HDI Rank (2018)', 'inform_risk', 'inform_p2p_hazard_and_exposure_dimension',
                   'population_density', 'population_living_in_urban_areas',
                   'proportion_of_population_with_basic_handwashing_facilities_on_premises',
                   'people_using_at_least_basic_sanitation_services',
                   'inform_vulnerability', 'inform_health_conditions',
                   'inform_epidemic_vulnerability', 'mortality_rate_under_5',
                   'prevalence_of_undernourishment', 'inform_lack_of_coping_capacity',
                   'inform_access_to_healthcare', 'current_health_expenditure_per_capita',
                   'maternal_mortality_ratio']


# Plotting cluster to world map
def plot_clusters(df_k, title):
    colorscale = [[0, 'blue'], [0.25, 'green'], [0.5, 'yellow'], [0.75, 'orange'], [1, 'red']]

    map_data = [dict(type='choropleth',
                 locations=df_k['country_region'].astype(str),
                 z=df_k['cluster'].astype(int),
                 locationmode='country names',
                 colorscale=colorscale)]

    final_map = dict(data=map_data,
               layout_title_text="<b>" + title + "</b>")
    plotly.offline.plot(final_map)


# Plotting all world maps together
def plot_multiple_maps(df_list, title=None):
    ## plot result
    _colorscale = [[0, 'blue'], [0.25, 'green'], [0.5, 'yellow'], [0.75, 'orange'], [1, 'red']]
    ROW, COL = 3, 1
    if not title: title = 'Unscaled vs Scaled vs Scaled with Top Factors'
    fig = tls.make_subplots(rows=ROW, cols=COL, column_widths=[1], row_heights=[0.33, 0.33, 0.33],
                            specs=[[{"type": "choropleth"}], [{"type": "choropleth"}], [{"type": "choropleth"}]])

    for r in range(ROW):
        for c in range(COL):
            _df = df_list[c * ROW + r]
            fig.add_trace(
                go.Choropleth(type='choropleth',
                              locations=_df['country_region'].astype(str),
                              z=_df['cluster'].astype(int),
                              locationmode='country names',
                              showscale=True, colorscale=_colorscale,
                              colorbar=dict(
                                  title="Cluster Index",
                                  yanchor="top", x=-0.2, y=1,
                                  ticks="outside", ticksuffix="(num)",
                              ), ),
                row=r + 1, col=c + 1
            )

    fig.update_layout(
        title=title,
        autosize=True,
        width=1400,
        height=900,
    )

    fig.show()


dataset_feature = pd.read_csv(INPUT_FILE)

not_null_features_df = dataset_feature[dataset_feature[NULL_FEATURES].notnull().all(1)]
not_zero_total_tests_df = not_null_features_df.loc[not_null_features_df['total_covid_19_tests'] != 0]
dataset_features_by_country = not_zero_total_tests_df[FILTER_FEATURES]
dataset_features_by_country.fillna(0)

dataset_features_by_country.loc[
    dataset_features_by_country.Country_Region == 'US', 'Country_Region'] = 'United States of America'

temp_data = dataset_features_by_country.sort_values(by=["Country_Region"])
temp_data = temp_data.reset_index(drop=True)

temp_data['pop2020'] = temp_data['pop2020'].apply(lambda x: x * 1000)
temp_data["confirmed_ratio"] = temp_data["Confirmed"] / temp_data["pop2020"]
temp_data["confirmed_ratio"] = temp_data["confirmed_ratio"].apply(lambda x: x * 1000)

temp_data["test_ratio"] = temp_data["total_covid_19_tests"] / temp_data["pop2020"]
temp_data["test_ratio"] = temp_data["test_ratio"].apply(lambda x: x * 1000)
temp_data = temp_data.replace("No data", 0)
temp_data = temp_data.replace(np.inf, 0)
temp_data = temp_data.replace(np.nan, 0)
temp_data = temp_data.replace('x', 0)

print(temp_data)

# Plot confirmed cases on the world map
colorscale = [[0, 'blue'], [0.25, 'green'], [0.5, 'yellow'], [0.75, 'orange'], [1, 'red']]
world_map = [dict(type='choropleth',
             locations=temp_data['Country_Region'].astype(str),
             z=temp_data['Confirmed'].astype(int),
             locationmode='country names',
             colorscale=colorscale)]

final_map = dict(data=world_map,
           layout_title_text="<b>Confirmed COVID-19 Cases</b>")
plotly.offline.plot(final_map)

indicator_data = temp_data.drop(columns=["Country_Region", "pop2020", "Confirmed", "total_covid_19_tests"])
print("DATA FOR CLUSTERING\n", indicator_data.tail(10))
print("\nfeatures:", indicator_data.columns)

# ------------------------------------------------------------------------------------------
# CLUSTER WITH UNSCALED DATA
# ------------------------------------------------------------------------------------------
data_unscaled = temp_data.drop(columns=["Country_Region", "pop2020", "Confirmed", "total_covid_19_tests"])

# Plot inertia to find the best number of clusters to use
inertia_elbow = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(data_unscaled)
    inertia_elbow.append(kmeans.inertia_)
plt.plot(range(1, 15), inertia_elbow)
plt.xlabel("# of clusters")
plt.ylabel("Inertia")
plt.title("Optimal clusters for UNSCALED data")
plt.show()

# Find the factor that impacts the confirmed ratio the most to visualize the clusters
mutual_info = mutual_info_regression(indicator_data.drop(columns=['confirmed_ratio']), indicator_data['confirmed_ratio'])
mutual_info = pd.Series(mutual_info)
mutual_info.index = indicator_data.drop(columns=['confirmed_ratio']).columns
mutual_info.sort_values(ascending=False)
mutual_info.sort_values(ascending=False).plot.bar(figsize=(10, 4))
plt.title("Factors impacting COVID-19 confirmed cases ratio")
plt.show()


# Cluster without scaling
kmeans = KMeans(n_clusters=5, init='k-means++')
kmeans.fit(data_unscaled)
predicted_cluster = kmeans.predict(data_unscaled)

data_unscaled["country_region"] = temp_data["Country_Region"]
data_unscaled['cluster'] = predicted_cluster
print("\nDATAFRAME WITHOUT SCALING")
print(data_unscaled.tail(10))
print("\nCluster counts:")
print(data_unscaled['cluster'].value_counts())

# Call plot_clusters function to plot clusters with unscaled_data
plot_clusters(data_unscaled, title="Clusters With UnScaled Data Based On All Factors")

print("\nCLUSTERS WITHOUT SCALING")
for clustering in range(0, 5):
    countries = data_unscaled.loc[data_unscaled['cluster'] == clustering]
    clustered_countries_list = list(countries['country_region'])
    print("Group", clustering, ":", clustered_countries_list, "\n-------------------")

# Plot cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(data_unscaled['current_health_expenditure_per_capita'], data_unscaled["confirmed_ratio"], c=predicted_cluster,
            cmap='rainbow')
ax = plt.gca()
ax.set_xticks([20, 40, 60, 80])
ax.set_xticklabels(['20', '40', '60', '80'])
plt.title('Covid Clustering for UNSCALED DATA')
plt.xlabel("Current Health Expenditure Per Capita")
plt.ylabel("Ratio of Confirmed COVID Cases")
plt.show()

data_unscaled['HDI Rank (2018)'] = data_unscaled['HDI Rank (2018)'].astype(float)
data_unscaled['current_health_expenditure_per_capita'] = data_unscaled['current_health_expenditure_per_capita'].astype(
    float)
data_unscaled['mortality_rate_under_5'] = data_unscaled['mortality_rate_under_5'].astype(float)

df_cluster = data_unscaled[
    ['cluster', 'confirmed_ratio', 'current_health_expenditure_per_capita', 'test_ratio', 'inform_risk',
     'HDI Rank (2018)', 'mortality_rate_under_5']]
cluster_avgs = pd.DataFrame(round(df_cluster.groupby('cluster').mean(), 1))
print("\nCLUSTER UNSCALED AVERAGES\n", cluster_avgs)
print("===========================")

# ------------------------------------------------------------------------------------------
# CLUSTER WITH SCALED DATA
# ------------------------------------------------------------------------------------------
scaler = StandardScaler()
print("DATA FOR CLUSTERING\n", indicator_data.tail(10))
data_k = scaler.fit_transform(indicator_data)

# Plot inertia to find the best number of clusters to use
inertia = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(data_k)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 15), inertia)
plt.xlabel("# of clusters")
plt.ylabel("Inertia")
plt.title("Optimal clusters for SCALED data")
plt.show()

# Cluster based on ALL features
kmeans = KMeans(n_clusters=5, init='k-means++')
kmeans.fit(data_k)
predicted_cluster = kmeans.predict(data_k)

# Convert scaled matrix back into dataframe and add in column names
df_k = pd.DataFrame(data_k)
df_k.columns = indicator_data.columns
df_k["country_region"] = temp_data["Country_Region"]
df_k['cluster'] = predicted_cluster
print("\nDATAFRAME WITH SCALING")
print(df_k.tail(10))
print("\nCluster counts:")
print(df_k['cluster'].value_counts())

print("\nCLUSTERS WITH SCALING")
for clustering in range(0, 5):
    countries = df_k.loc[df_k['cluster'] == clustering]
    clustered_countries_list = list(countries['country_region'])
    print("Group", clustering, ":", clustered_countries_list, "\n-------------------")

# call plot_clusters function to plot clusters with scaled_data
plot_clusters(df_k, title="Clusters With Scaled Data Based On All Factors")

# Plot cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(df_k['current_health_expenditure_per_capita'], df_k["confirmed_ratio"], c=predicted_cluster, cmap='rainbow')
plt.title('Covid Clustering for SCALED DATA')
plt.xlabel("Current Health Expenditure Per Capita")
plt.ylabel("Ratio of Confirmed COVID Cases")
plt.show()

# Find cluster averages
df_cluster = df_k[['cluster', 'confirmed_ratio', 'current_health_expenditure_per_capita', 'test_ratio', 'inform_risk',
                   'HDI Rank (2018)', 'mortality_rate_under_5']]
cluster_avgs = pd.DataFrame(round(df_cluster.groupby('cluster').mean(), 1))
print("\nCLUSTER SCALED AVERAGES\n", cluster_avgs)
print("===========================")

# ------------------------------------------------------------------------------------------
# CLUSTER WITH TOP FACTORS & SCALED DATA
# ------------------------------------------------------------------------------------------
df_top = df_cluster.drop(columns=['cluster'])
kmeans = KMeans(n_clusters=5, init='k-means++')
kmeans.fit(df_top)
predicted_cluster = kmeans.predict(df_top)

df_top["country_region"] = temp_data["Country_Region"]
df_top['cluster'] = predicted_cluster
print("\nDATAFRAME SCALED TOP FACTORS")
print(df_top.tail(10))
print("\nCluster counts:")
print(df_top['cluster'].value_counts())

for clustering in range(0, 5):
    countries = df_top.loc[df_top['cluster'] == clustering]
    clustered_countries_list = list(countries['country_region'])
    print("Group", clustering, ":", clustered_countries_list, "\n")

# Call plot_clusters function to plot clusters with top k data
plot_clusters(df_top, title="5 Clusters Based On Top 5 Factors")

# Plot cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(df_top['current_health_expenditure_per_capita'], df_top["confirmed_ratio"], c=predicted_cluster, cmap='rainbow')

plt.title('Covid Clustering for TOP FACTORS AND SCALED DATA')
plt.xlabel("Current Health Expenditure Per Capita")
plt.ylabel("Ratio of Confirmed COVID Cases")
plt.show()

cluster_avgs = pd.DataFrame(round(df_top.groupby('cluster').mean(), 1))
print("\nCLUSTER SCALED TOP AVERAGES\n", cluster_avgs)

#call plot_multiple_maps function
plot_multiple_maps([data_unscaled, df_k, df_top], title=None)
