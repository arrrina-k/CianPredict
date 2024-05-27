import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

features = ["author_type", "floor", "floors_count", "rooms_count", "total_meters", "district", "underground"]


@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    model = CatBoostRegressor()
    model.load_model(fname="./weights/catboost_model.pth")
    return model


@st.cache_data(show_spinner="Loading data...")
def load_data():
    data = pd.read_csv("./data/processed/scraped_cian_processed.csv")
    return data


def predict(model: CatBoostRegressor):

    data = pd.DataFrame(
        {key: value for key, value in st.session_state.items() if key != "FormSubmitter:Data to predict-Predict"},
        index=[0],
    )
    data = data[features]

    return model.predict(data)


if __name__ == "__main__":
    st.header("This is a model inferencing section 🪄")
    st.subheader("We've considered to train 4 models. Let's checkout the results.")

    st.divider()

    st.subheader(
        " 1. First approach was to fit the model with all categorical features indeed. They were introduced as one-hot-encoding labels."
    )
    code_model1 = """
        df_processed = pd.get_dummies(df, columns=['author_type','underground', 'district','street'])

        X = df_processed.drop(columns=['price', 'lats', 'lons'], axis=1)
        y = df_processed['price']

        X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
    """

    st.code(code_model1, language="python")

    st.subheader("The result was represented by the following metrics:")
    st.markdown(
        """
        $$
        r2: 0.69
        $$
        $$
        MSE: 25.55
        $$
    """
    )

    st.divider()

    st.subheader(
        " 2. After that we considered to remove ‘street’ feature, because the number of unique examples on each street hardly reached 5."
    )

    code_model2 = """
        df_processed_without_streets = pd.get_dummies(df, columns=['author_type', 'district', 'underground'])

        X = df_processed_without_streets.drop(columns=['price','street','lats', 'lons'])
        Y = df_processed_without_streets['price']

        X_train_streets, X_test_streets, Y_train_streets, Y_test_streets = train_test_split(X, Y, random_state=42, test_size=0.2)

        model_streets = LinearRegression()
        model_streets.fit(X_train_streets, Y_train_streets)
        Y_pred_streets = model_streets.predict(X_test_streets)

        mse_streets = mean_squared_error(Y_test_streets, Y_pred_streets)
        r2_score_streets = r2_score(Y_test_streets, Y_pred_streets)
    """

    st.code(code_model2, language="python")
    st.subheader("The results reached:")
    st.markdown(
        """
        $$
        r2: 0.78
        $$
        $$
        MSE: 18.06
        $$
    """
    )

    st.divider()
    st.subheader(" 3. Then we tried to preprocess categorical features by mean target encoding.")

    code_model3 = """
        categorical = ['author_type', 'underground', 'district']

        df_mean_target = df.drop(columns=['street','lats', 'lons'])
        for feature in categorical:
            mean_prices = df_mean_target.groupby(feature)['price'].mean()
            df_mean_target[feature] = df_mean_target[feature].map(mean_prices)

        X_mean_target = df_mean_target.drop(columns='price')
        Y_mean_target = df_mean_target['price']

        X_train_mean_target, X_test_mean_target,Y_train_mean_target, Y_test_mean_target = train_test_split(X_mean_target, Y_mean_target, test_size=0.2, shuffle=True)

        model_mean_target = LinearRegression()
        model_mean_target.fit(X_train_mean_target, Y_train_mean_target)
        Y_pred_mean_target = model_mean_target.predict(X_test_mean_target)

        mse_mean_target = mean_squared_error(Y_test_mean_target, Y_pred_mean_target)
        r2_score_mean_target = r2_score(Y_test_mean_target, Y_pred_mean_target)
    """
    st.code(code_model3, language="python")
    st.subheader("The results didn't change:")
    st.markdown(
        """
        $$
        r2: 0.77
        $$
        $$
        MSE: 19.46
        $$
    """
    )

    st.divider()

    st.subheader(
        " 4. After that it was interesting for us to use another regressor. Such type of model becomes Yandex Catboost Regressor which can preprocess categorical features by itself."
    )
    code_model4 = """
        X_catboost = df.drop(columns=['price', 'lats', 'lons', 'street'])
        Y_catboost = df['price']

        X_train_catboost, X_test_catboost, Y_train_catboost, Y_test_catboost = train_test_split(X_catboost, Y_catboost, random_state=42, test_size=0.2)

        catboost_model = CatBoostRegressor()
        catboost_model.fit(X_train_catboost, Y_train_catboost, cat_features=categorical)

        Y_predict_catboost = catboost_model.predict(X_test_catboost)
        mse_catboost = mean_squared_error(Y_test_catboost, Y_predict_catboost)
        r2_catboost = r2_score(Y_test_catboost, Y_predict_catboost)
    """
    st.code(code_model4, language="python")
    st.subheader("Checkout the metrics: ")
    st.markdown(
        """
        $$
        r2: 0.85
        $$
        $$
        MSE: 12.73
        $$
    """
    )

    st.divider()

    st.header("Live Inferencing ✨")
    st.subheader("Here you can fill out the form and predict the price of your apartment with the best Catboost Model")

    model = load_model()
    data = load_data()
    with st.form("Data to predict"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.selectbox("Author ✍️", (sorted(data["author_type"].unique())), key="author_type")

        with col2:
            st.number_input("Apartment's Floor 🦶🏼", min_value=-1, max_value=100, value=1, key="floor")

        with col3:
            st.number_input("Acocommodation Floors Count 🏠", min_value=0, max_value=100, value=1, key="floors_count")

        col4, col5, col6 = st.columns(3)

        with col4:
            st.number_input("Apartment's Room Count 🚪", min_value=1, max_value=30, value=1, key="rooms_count")

        with col5:
            st.number_input("Apartment's Square 📐", min_value=1, max_value=500, value=10, key="total_meters")

        with col6:
            st.selectbox("District 🏙", (sorted(data["district"].unique())), key="district")

        col7 = st.columns(1)[0]

        with col7:
            st.selectbox("Underground 🚇", (sorted(data["underground"].unique())), key="underground")

        click = st.form_submit_button("Predict")

    if click:
        res = predict(model)
        st.write("Accommodation price : ", round(float(res[0]), 2), "million rubles 💰")
        st.balloons()
