import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import scipy.stats as stats
import streamlit as st
from dython.nominal import associations

# Helper section
price_ranges = {
    "0-17.5 million rubles": [20, 15, 150],
    "17.5-27.0 million rubles": [200, 10, 50],
    "27+ million rubles": [0, 0, 0],
}


def color_select(x, data):
    mean_underground = data.groupby(["underground"])["price"].mean()
    quantile_25 = mean_underground.quantile(0.25)
    quantile_75 = mean_underground.quantile(0.75)

    if x <= quantile_25:
        return [20, 15, 150, 200]
    elif quantile_25 < x <= quantile_75:
        return [200, 10, 50, 200]
    else:
        return [0, 0, 0, 200]


# Data caching
@st.cache_data(show_spinner="Loading data...")
def load_data():
    data = pd.read_csv("data/processed/scraped_cian_processed.csv")
    data_3d = data.groupby(["underground", "lats", "lons"])["price"].mean().reset_index()
    data_3d["price"] = data_3d["price"].round(2)
    data_3d["color"] = data_3d["price"].apply(color_select, args=[data])
    return data_3d, data


# Pydeck initialization for 3D map
@st.cache_data(show_spinner="Setting configuration...")
def setting_configuration(data):
    view_state = pdk.ViewState(latitude=55.7520233, longitude=37.6174994, zoom=10, bearing=0, pitch=45)
    layer = pdk.Layer(
        "ColumnLayer",
        data=data,
        get_position=["lons", "lats"],
        get_elevation=["price"],
        radius=300,
        auto_highlight=True,
        elevation_scale=100,
        pickable=True,
        elevation_range=[0, 3000],
        get_fill_color="color",
        extruded=True,
    )

    return view_state, layer


if __name__ == "__main__":
    st.header("Exploratory Data Analysis 🗺️")

    # -- Loading data section
    data_3d, data = load_data()
    view_state, layer = setting_configuration(data_3d)
    # --
    st.divider()
    st.subheader("Let's overview our data in interactive way")

    # -- Multiselect data section
    st.multiselect(
        label="Here you can choose a set of columns to display from the whole dataframe. Note, that price is in million of rubles units.",
        options=set(data.columns),
        key="data_columns",
        default=["author_type", "underground", "total_meters", "price"],
    )
    st.dataframe(data[st.session_state["data_columns"]])
    # --
    st.divider()
    st.subheader("Check out 3D visualisation of flat mean price in Moscow for each metro station")
    st.subheader(
        "It's easy to see a cocenter circle distribution over three flat cost segments. The most expensive flats are located in the center etc."
    )

    # -- Legend section
    for label, color in price_ranges.items():
        st.markdown(
            f"<span style='display:inline-block; width: 12px; height: 12px; margin-right: 5px; background-color: rgb{tuple(color)};'></span>{label}",
            unsafe_allow_html=True,
        )
    # --
    # -- 3D Map section
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "text": "Price: {price} mil rubles, Metro: {underground}",
            },
            map_provider="mapbox",
            map_style=pdk.map_styles.MAPBOX_ROAD,
        )
    )
    # --
    st.divider()
    st.subheader("In the next step we need to see all the unique values of categorical features")

    # -- Uniques section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Author Type Unique Values")
        st.write(data["author_type"].unique())
    with col2:
        st.write("Unique District Values")
        st.write(data["district"].unique())
    with col3:
        st.write("Unique Underground Values")
        st.write(data["underground"].unique())
    # --
    st.divider()
    st.subheader("Let's take a glance on a different categorical feature distribution among Mean Price")
    features = ["underground", "district", "floor", "total_meters", "rooms_count", "author_type"]

    # -- Charts over Mean Price section
    for feature in features:
        st.line_chart(
            pd.DataFrame(
                {
                    feature: sorted(data[feature].unique().tolist()),
                    "Mean Price (Million Rubles Unit)": data.groupby([feature]).mean("price")["price"],
                }
            ),
            y="Mean Price (Million Rubles Unit)",
            x=feature,
        )
    # --
    st.subheader(
        "So, we can see that Mean Price Distribution among underground and district features has very noisy graphics."
        " We cann't estimate some useful information from it."
        " But we can see, that mean price of flats on lower floors, with small square and small rooms count much cheeper."
        " Also need to be said, that there is a huge difference between authors of advertisments."
        " All these features except underground and district, has a significant impact 👉 we need to perform additional hypotesis estimation of them on independence for price."
    )
    st.divider()
    st.subheader("(We considered to use ANOVA tests because we want to compare the means of more than two groups)")
    underground_groups = [
        data[data["underground"] == station]["price"].values
        for station in data["underground"].unique()
        if len(data[data["underground"] == station]) > 1
    ]
    f_val, p_val = stats.f_oneway(*underground_groups)
    st.subheader(
        "Hypothesis Test with **\u03B1** = 0.05 which will estimate the significance of the impact of underground station area on price👇"
    )
    st.markdown(
        "**H0:** There is no difference in the mean property prices across different underground station areas."
    )
    st.markdown(
        "**H1:**  At least one underground station area has a mean property price that is different from the others."
    )
    st.subheader("ANOVA Test Results")
    st.write("F-value: {:.2f}".format(f_val))
    st.write("p-value: {:.2f}".format(p_val))
    if p_val < 0.05:
        st.write(
            "Conclusion: **p_value < \u03B1** which means that we reject the null hypothesis (H0). There are significant differences in prices across different underground station areas."
        )
    else:
        st.write(
            "Conclusion: **p_value > \u03B1** which means that we fail to reject the null hypothesis (H0). There are no significant differences in prices across different underground station areas."
        )
    st.divider()
    street_groups = [
        data[data["street"] == street]["price"].values
        for street in data["street"].unique()
        if len(data[data["street"] == street]) > 1
    ]
    f_val, p_val = stats.f_oneway(*street_groups)
    st.subheader(
        "Now we should conduct the same statistical test which will estimate the significance of the impact of streets on price👇"
    )
    st.markdown("**H0:** There is no difference in the mean prices across different streets.")
    st.markdown("**H1:** At least one street has a mean price that is different from the others.")
    st.subheader("ANOVA Test Results")
    st.write("F-value: {:.2f}".format(f_val))
    st.write("p-value: {:.2f}".format(p_val))
    if p_val < 0.05:
        st.write(
            "Conclusion: **p_value < \u03B1** which means that we reject the null hypothesis (H0). There are significant differences in prices across different streets."
        )
    else:
        st.write(
            "Conclusion: **p_value > \u03B1** which means that we fail to reject the null hypothesis (H0). There are no significant differences in prices across different streets."
        )
    st.divider()
    st.subheader(
        "Also, we need to check the hypothesis that official_representative author mean price higher than representative developer mean_price. It'll be better to use t-test for this case. (\u03B1 remains the same)"
    )
    official_representative_prices = data[data["author_type"] == "Агент по недвижимости"]["price"]
    developer_prices = data[data["author_type"] == "Застройщик"]["price"]
    t_stat, p_value = stats.ttest_ind(
        official_representative_prices, developer_prices, alternative="greater", equal_var=False
    )
    st.markdown(
        "**H0:** There is no difference in the mean prices between listings by official representatives and developers."
    )
    st.markdown("**H1:**  The mean price of listings by official_representative is higher than those by developer.")
    st.subheader("ANOVA Test Results")
    st.write("t-statistic: {:.2f}".format(t_stat))
    st.write("p-value: {:.2f}".format(p_val))
    if p_val < 0.05:
        st.write(
            "Conclusion: **p_value < \u03B1** which means that we reject the null hypothesis (H0). The mean price of listings by official_representative is higher than those by developers."
        )
    else:
        st.write(
            "Conclusion: **p_value > \u03B1** which means that we fail to reject the null hypothesis (H0). There is no significant difference in mean property prices between official representatives and developers."
        )

    st.divider()
    st.subheader("Price box plot section")
    st.subheader("Here, we have tried to check out outlier values, but there are none")
    st.subheader(
        "Also, we need to mention, that 75% of data is under 30 million of rubles and 25% of data is under 15 millions. Median value is 22 millions approximately."
    )

    # -- Box plot price section
    st.altair_chart(
        alt.Chart(data)
        .mark_boxplot(color="#5a88ae")
        .encode(
            x=alt.X("price", sort=None, title="price"),
        )
    )
    st.divider()
    # --
    st.subheader("Here we can see linear correlation feature-feature matrix")
    st.subheader(
        "In the recent sections we tried to estimate some correlation between Distric and Underground on Mean Price, but noise in graphics bothered us. So, in this correlation matrix we can see that correlation exists! But again, we also need to check the hypotesis and count of unique values in our data. Correlation between other features confirms"
    )

    # -- Correlation matrix section
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#0d1118")
    associations(data, ax=ax, figsize=(10, 10))

    ax.tick_params(axis="both", colors="#5a88ae")

    st.pyplot(fig)

    # --

    st.subheader(
        "There is an interesting feature that street heavily impacts on the price and other features. Let's dive into and see how many unique streets we have."
    )
    st.divider()

    st.subheader("Diving deeper")
    st.subheader(
        "We can see that the majority of all streets apperas about 1-5 times, thus it's evidently that there is no correlation between price and street. Further we need to delete this feature from our dataframe."
    )

    # -- Underground/Street/District count plotting section

    # Using st.altair_chart to visualise graphics, because there is an issue with origin st.line_chart which sorts data on x-axis automatically

    features = ["street", "district", "underground"]

    for feature in features:
        df = data[feature].value_counts().reset_index()
        st.altair_chart(
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X(feature, sort=None, title=feature),
                y=alt.Y("count", title="count"),
            ),
            use_container_width=True,
        )
    # --

    st.subheader(
        "But, in District and Underground count plots values appear more than 5 times almost for each value 👉 correlation might be exist."
    )
