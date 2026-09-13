import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import time

st.set_page_config(
    page_title="AutoScope",
    page_icon="🚘",
    layout="wide"
)

#dataset
df = pd.read_csv("data/processed/used_car_dataset_cleaned.csv")

artifact = joblib.load(
    "models/car_price_model.pkl"
)

model = artifact["model"]
encoder = artifact["encoder"]

categorical_cols = [
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type"
]

numeric_cols = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

#sidebar
st.sidebar.title("AutoScope")

page = st.sidebar.radio(
    "Navigate to",
    [
        "🏠 Home",
        "📊 Data Insights",
        "💰 Price Prediction",
        "⚖️ Compare Cars"
    ]
)

#Home page
if page == "🏠 Home":
    st.title("Welcome to AutoScope! 🚘")
    
    st.markdown(
        "### Used Car Analytics, Price Prediction & Comparison"
    )

    st.write(
        "AutoScope is a data-driven application that explores "
        "the used-car market, analyzes pricing patterns, "
        "predicts car prices, and helps users compare cars."
    )

    st.divider()
    
    #Project highlights
    st.subheader("🔍 What can you do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Analyze")
        st.write(
            "Explore price trends, brands, fuel types, "
            "transmission and other car characteristics."
        )
        
    with col2:
        st.markdown("### 💰 Predict")
        st.write(
            "Estimate the selling price of a used car "
            "using a Linear Regression model."
        )
    
    with col3:
        st.markdown("### ⚖️ Compare")
        st.write(
            "Compare two cars across price, age, mileage, "
            "power, engine and other features."
        )

    st.divider()
    
    #Project information
    st.subheader("📌 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cars", "15K+")
    with col2:
        st.metric("Features", "12")
    with col3:
        st.metric("Model", "Regression")
    with col4:
        st.metric("R² Score", "72.8%")
    
#Data Analytics page
elif page == "📊 Data Insights":
    st.title("📊 Used Car Market Insights")

    st.write(
        "Explore the used-car market and understand how "
        "different car characteristics relate to price."
    )

    st.divider()

    #Market Overview
    st.subheader("📌 Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Cars",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Median Price",
            f"₹{df['selling_price'].median():,.0f}"
        )

    with col3:
        st.metric(
            "Brands",
            df["brand"].nunique()
        )

    with col4:
        st.metric(
            "Models",
            df["model"].nunique()
        )

    st.divider()

    #Car Price Summary
    st.subheader("💰 Price Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Lowest Price",
            f"₹{df['selling_price'].min():,.0f}"
        )

    with col2:
        st.metric(
            "Median Price",
            f"₹{df['selling_price'].median():,.0f}"
        )

    with col3:
        st.metric(
            "Average Price",
            f"₹{df['selling_price'].mean():,.0f}"
        )

    with col4:
        st.metric(
            "Highest Price",
            f"₹{df['selling_price'].max():,.0f}"
        )

    #Brand Analysis
    st.divider()

    st.subheader("🏷️ Brand Analysis")

    col1, col2 = st.columns(2)

    #Top brands by listings
    with col1:
        brand_counts = (
            df["brand"]
            .value_counts()
            .head(10)
            .sort_values()
        )

        fig_brand_count = px.bar(
            brand_counts,
            x=brand_counts.values,
            y=brand_counts.index,
            orientation="h",
            title="Top 10 Brands by Number of Cars",
            labels={
                "x": "Number of Cars",
                "y": "Brand"
            }
        )

        fig_brand_count.update_layout(
            height=500
        )

        st.plotly_chart(
            fig_brand_count,
            use_container_width=True
        )

    #Median price by brand
    with col2:

        top_brands = (
            df["brand"]
            .value_counts()
            .head(10)
            .index
        )

        brand_price = (
            df[df["brand"].isin(top_brands)]
            .groupby("brand")["selling_price"]
            .median()
            .sort_values()
        )

        fig_brand_price = px.bar(
            brand_price,
            x=brand_price.values,
            y=brand_price.index,
            orientation="h",
            title="Median Price of Top 10 Brands",
            labels={
                "x": "Median Selling Price (₹)",
                "y": "Brand"
            }
        )

        fig_brand_price.update_layout(
            height=500,
            xaxis_tickformat=",.0f"
        )

        st.plotly_chart(
            fig_brand_price,
            use_container_width=True
        )

    #Fuel Type
    st.divider()

    st.subheader("⛽ Fuel Type Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fuel_counts = (
            df["fuel_type"]
            .value_counts()
            .reset_index()
        )

        fuel_counts.columns = ["Fuel Type", "Cars"]

        fig_fuel_count = px.bar(
            fuel_counts,
            x="Fuel Type",
            y="Cars",
            title="Cars by Fuel Type",
            text="Cars"
        )

        fig_fuel_count.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_fuel_count,
            use_container_width=True
        )

    with col2:

        fuel_price = (
            df.groupby("fuel_type")["selling_price"]
            .median()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_fuel_price = px.bar(
            fuel_price,
            x="fuel_type",
            y="selling_price",
            title="Median Price by Fuel Type",
            labels={
                "fuel_type": "Fuel Type",
                "selling_price": "Median Selling Price (₹)"
            },
            text="selling_price"
        )

        fig_fuel_price.update_layout(
            height=450,
            yaxis_tickformat=",.0f"
        )

        fig_fuel_price.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig_fuel_price,
            use_container_width=True
        )

    #Transmission
    st.divider()

    st.subheader("⚙️ Transmission Analysis")

    col1, col2 = st.columns(2)

    with col1:

        transmission_counts = (
            df["transmission_type"]
            .value_counts()
            .reset_index()
        )

        transmission_counts.columns = [
            "Transmission",
            "Cars"
        ]

        fig_transmission_count = px.bar(
            transmission_counts,
            x="Transmission",
            y="Cars",
            title="Cars by Transmission Type",
            text="Cars"
        )

        fig_transmission_count.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_transmission_count,
            use_container_width=True
        )

    with col2:

        transmission_price = (
            df.groupby("transmission_type")["selling_price"]
            .median()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_transmission_price = px.bar(
            transmission_price,
            x="transmission_type",
            y="selling_price",
            title="Median Price by Transmission",
            labels={
                "transmission_type": "Transmission",
                "selling_price": "Median Selling Price (₹)"
            },
            text="selling_price"
        )

        fig_transmission_price.update_layout(
            height=450,
            yaxis_tickformat=",.0f"
        )

        fig_transmission_price.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig_transmission_price,
            use_container_width=True
        )

    #Engine and Power
    st.divider()

    st.subheader("⚡ Performance & Price")

    col1, col2 = st.columns(2)

    with col1:

        fig_engine = px.scatter(
            df,
            x="engine",
            y="selling_price",
            opacity=0.5,
            title="Engine Size vs Selling Price",
            labels={
                "engine": "Engine (cc)",
                "selling_price": "Selling Price (₹)"
            },
            hover_data=[
                "brand",
                "model",
                "vehicle_age"
            ]
        )

        fig_engine.update_layout(
            height=500,
            yaxis_tickformat=",.0f"
        )

        st.plotly_chart(
            fig_engine,
            use_container_width=True
        )

    with col2:

        fig_power = px.scatter(
            df,
            x="max_power",
            y="selling_price",
            opacity=0.5,
            title="Max Power vs Selling Price",
            labels={
                "max_power": "Max Power",
                "selling_price": "Selling Price (₹)"
            },
            hover_data=[
                "brand",
                "model",
                "vehicle_age"
            ]
        )

        fig_power.update_layout(
            height=500,
            yaxis_tickformat=",.0f"
        )

        st.plotly_chart(
            fig_power,
            use_container_width=True
        )

    #Seller Type
    st.divider()

    st.subheader("🏪 Seller Type")

    col1, col2 = st.columns(2)

    with col1:

        seller_counts = (
            df["seller_type"]
            .value_counts()
            .reset_index()
        )

        seller_counts.columns = [
            "Seller Type",
            "Cars"
        ]

        fig_seller_count = px.bar(
            seller_counts,
            x="Seller Type",
            y="Cars",
            title="Cars by Seller Type",
            text="Cars"
        )

        fig_seller_count.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_seller_count,
            use_container_width=True
        )

    with col2:

        seller_price = (
            df.groupby("seller_type")["selling_price"]
            .median()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_seller_price = px.bar(
            seller_price,
            x="seller_type",
            y="selling_price",
            title="Median Price by Seller Type",
            labels={
                "seller_type": "Seller Type",
                "selling_price": "Median Selling Price (₹)"
            },
            text="selling_price"
        )

        fig_seller_price.update_layout(
            height=450,
            yaxis_tickformat=",.0f"
        )

        fig_seller_price.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig_seller_price,
            use_container_width=True
        )

    #Conclusion
    st.divider()

    st.subheader("💡 Key Takeaways")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **🚘 Vehicle Age**
            
            Cars generally become less expensive as vehicle age
            increases.

            **🏷️ Brand**
            
            Maruti has the highest number of listings, while BMW
            has a much higher typical selling price.

            **⚡ Performance**
            
            Engine size and especially maximum power show a
            positive relationship with selling price.
            """
        )

    with col2:

        st.markdown(
            """
            **⛽ Fuel**
            
            Petrol and diesel dominate the dataset. Very small
            categories such as electric should be interpreted
            carefully.

            **⚙️ Transmission**
            
            Automatic cars have a considerably higher median
            price than manual cars in this dataset.

            **🏪 Seller Type**
            
            Dealer listings make up the largest portion of the
            dataset.
            """
        )
    
#Car price prediction page
elif page == "💰 Price Prediction":
    st.title("💰 Used Car Price Prediction")

    st.write(
        "Enter the details of a used car to estimate its selling price."
    )

    st.divider()

    #Feature columns used by the model
    categorical_cols = [
        "brand",
        "model",
        "seller_type",
        "fuel_type",
        "transmission_type"
    ]

    numeric_cols = [
        "vehicle_age",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "seats"
    ]

    #car details
    st.subheader("🚘 Car Details")

    col1, col2 = st.columns(2)

    with col1:

        brands = sorted(
            df["brand"].dropna().unique()
        )

        brand = st.selectbox(
            "Brand",
            brands
        )

    with col2:

        models = sorted(
            df[df["brand"] == brand]["model"]
            .dropna()
            .unique()
        )

        model_name = st.selectbox(
            "Model",
            models
        )

    st.divider()

    #Vehicle information
    st.subheader("📋 Vehicle Information")

    col1, col2 = st.columns(2)

    with col1:

        vehicle_age = st.number_input(
            "Vehicle Age (Years)",
            min_value=int(df["vehicle_age"].min()),
            max_value=int(df["vehicle_age"].max()),
            value=5,
            step=1
        )

        km_driven = st.number_input(
            "KM Driven",
            min_value=int(df["km_driven"].min()),
            max_value=int(df["km_driven"].max()),
            value=50000,
            step=1000
        )

        mileage = st.number_input(
            "Mileage",
            min_value=float(df["mileage"].min()),
            max_value=float(df["mileage"].max()),
            value=float(df["mileage"].median()),
            step=0.1
        )

        engine = st.number_input(
            "Engine (cc)",
            min_value=int(df["engine"].min()),
            max_value=int(df["engine"].max()),
            value=int(df["engine"].median()),
            step=100
        )

    with col2:

        max_power = st.number_input(
            "Max Power",
            min_value=float(df["max_power"].min()),
            max_value=float(df["max_power"].max()),
            value=float(df["max_power"].median()),
            step=1.0
        )

        seats = st.number_input(
            "Seats",
            min_value=int(df["seats"].min()),
            max_value=int(df["seats"].max()),
            value=5,
            step=1
        )

        seller_type = st.selectbox(
            "Seller Type",
            sorted(
                df["seller_type"].dropna().unique()
            )
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(
                df["fuel_type"].dropna().unique()
            )
        )

    transmission_type = st.selectbox(
        "Transmission",
        sorted(
            df["transmission_type"].dropna().unique()
        )
    )

    st.divider()

    #Prediction button
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        predict_button = st.button(
            "💰 Predict Price"
        )

    #prediction
    if predict_button:

        with st.spinner(
            "🚗 Analyzing car details..."
        ):

            # Small delay so the spinner is visible
            time.sleep(1.2)

            # Create one-row DataFrame
            input_data = pd.DataFrame({
                "brand": [brand],
                "model": [model_name],
                "vehicle_age": [vehicle_age],
                "km_driven": [km_driven],
                "seller_type": [seller_type],
                "fuel_type": [fuel_type],
                "transmission_type": [transmission_type],
                "mileage": [mileage],
                "engine": [engine],
                "max_power": [max_power],
                "seats": [seats]
            })

            #Encode categorical features
            input_cat = encoder.transform(
                input_data[categorical_cols]
            )

            #Combine numerical + categorical
            input_final = np.hstack([
                input_data[numeric_cols].values,
                input_cat
            ])

            #Predicted log price
            predicted_log_price = model.predict(
                input_final
            )[0]

            # Convert log prediction
            # back to original price
            predicted_price = np.expm1(
                predicted_log_price
            )

        #Result
        st.success(
            "Price prediction completed!"
        )

        st.subheader("💰 Estimated Car Price")

        col1, col2, col3 = st.columns(3)

        with col2:

            st.metric(
                "Estimated Selling Price",
                f"₹{predicted_price:,.0f}"
            )

        st.info(
            f"Approximately ₹{predicted_price / 100000:.2f} Lakh"
        )

elif page == "⚖️ Compare Cars":
    st.title("⚖️ Compare Cars")

    st.write(
        "Select two cars from the dataset and compare their "
        "price, condition, performance and specifications."
    )

    st.divider()

    # Create car selection labels
    car_options = df.index.tolist()

    def car_label(index):

        car = df.loc[index]

        return (
            f"{car['brand']} {car['model']} | "
            f"₹{car['selling_price']:,.0f} | "
            f"{car['vehicle_age']} yrs | "
            f"{car['km_driven']:,} km | "
            f"ID {index}"
        )

    # Select two cars
    st.subheader("🚗 Select Cars")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🚘 Car 1")

        car_1 = st.selectbox(
            "Choose the first car",
            car_options,
            format_func=car_label,
            key="car_1"
        )

    with col2:

        st.markdown("### 🚙 Car 2")

        car_2 = st.selectbox(
            "Choose the second car",
            car_options,
            format_func=car_label,
            index=1,
            key="car_2"
        )

    st.divider()

    # Get selected cars
    car_a = df.loc[car_1]
    car_b = df.loc[car_2]

    st.subheader("📋 Car Comparison")


    # Basic information
    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"## 🚘 {car_a['brand']} {car_a['model']}"
        )

        st.metric(
            "Selling Price",
            f"₹{car_a['selling_price']:,.0f}"
        )

    with col2:

        st.markdown(
            f"## 🚙 {car_b['brand']} {car_b['model']}"
        )

        st.metric(
            "Selling Price",
            f"₹{car_b['selling_price']:,.0f}"
        )

    st.divider()


    # Comparison table
    comparison = pd.DataFrame({
        "Specification": [
            "Brand",
            "Model",
            "Selling Price",
            "Vehicle Age",
            "KM Driven",
            "Mileage",
            "Engine",
            "Max Power",
            "Seats",
            "Fuel Type",
            "Transmission",
            "Seller Type"
        ],

        "Car 1": [
            car_a["brand"],
            car_a["model"],
            f"₹{car_a['selling_price']:,.0f}",
            f"{car_a['vehicle_age']} years",
            f"{car_a['km_driven']:,} km",
            f"{car_a['mileage']:.1f}",
            f"{car_a['engine']} cc",
            f"{car_a['max_power']:.1f}",
            int(car_a["seats"]),
            car_a["fuel_type"],
            car_a["transmission_type"],
            car_a["seller_type"]
        ],

        "Car 2": [
            car_b["brand"],
            car_b["model"],
            f"₹{car_b['selling_price']:,.0f}",
            f"{car_b['vehicle_age']} years",
            f"{car_b['km_driven']:,} km",
            f"{car_b['mileage']:.1f}",
            f"{car_b['engine']} cc",
            f"{car_b['max_power']:.1f}",
            int(car_b["seats"]),
            car_b["fuel_type"],
            car_b["transmission_type"],
            car_b["seller_type"]
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    #Visual comparision
    st.subheader("📊 Visual Comparison")

    chart_data = pd.DataFrame({
        "Car": [
            f"{car_a['brand']} {car_a['model']}",
            f"{car_b['brand']} {car_b['model']}"
        ],

        "Selling Price": [
            car_a["selling_price"],
            car_b["selling_price"]
        ],

        "Vehicle Age": [
            car_a["vehicle_age"],
            car_b["vehicle_age"]
        ],

        "KM Driven": [
            car_a["km_driven"],
            car_b["km_driven"]
        ],

        "Mileage": [
            car_a["mileage"],
            car_b["mileage"]
        ],

        "Engine": [
            car_a["engine"],
            car_b["engine"]
        ],

        "Max Power": [
            car_a["max_power"],
            car_b["max_power"]
        ]
    })

    #Price comparision
    fig_price = px.bar(
        chart_data,
        x="Car",
        y="Selling Price",
        text="Selling Price",
        title="💰 Selling Price Comparison"
    )

    fig_price.update_traces(
        texttemplate="₹%{text:,.0f}",
        textposition="outside"
    )

    fig_price.update_layout(
        height=450,
        yaxis_tickformat=",.0f"
    )

    st.plotly_chart(
        fig_price,
        use_container_width=True
    )

    #Performance comparision

    col1, col2 = st.columns(2)

    with col1:

        performance_data = pd.DataFrame({
            "Car": [
                f"{car_a['brand']} {car_a['model']}",
                f"{car_b['brand']} {car_b['model']}"
            ],

            "Engine": [
                car_a["engine"],
                car_b["engine"]
            ],

            "Max Power": [
                car_a["max_power"],
                car_b["max_power"]
            ]
        })

        fig_performance = px.bar(
            performance_data,
            x="Car",
            y=["Engine", "Max Power"],
            barmode="group",
            title="⚡ Engine & Power Comparison"
        )

        fig_performance.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_performance,
            use_container_width=True
        )

    with col2:

        usage_data = pd.DataFrame({
            "Car": [
                f"{car_a['brand']} {car_a['model']}",
                f"{car_b['brand']} {car_b['model']}"
            ],

            "Vehicle Age": [
                car_a["vehicle_age"],
                car_b["vehicle_age"]
            ],

            "KM Driven": [
                car_a["km_driven"],
                car_b["km_driven"]
            ]
        })

        fig_usage = px.bar(
            usage_data,
            x="Car",
            y=["Vehicle Age", "KM Driven"],
            barmode="group",
            title="📅 Age & Usage Comparison"
        )

        fig_usage.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_usage,
            use_container_width=True
        )

    st.divider()

    #Conclsion
    st.subheader("💡 What Stands Out?")

    # Price
    if car_a["selling_price"] < car_b["selling_price"]:
        cheaper = "Car 1"
        cheaper_name = f"{car_a['brand']} {car_a['model']}"
        price_difference = (
            car_b["selling_price"] - car_a["selling_price"]
        )
    else:
        cheaper = "Car 2"
        cheaper_name = f"{car_b['brand']} {car_b['model']}"
        price_difference = (
            car_a["selling_price"] - car_b["selling_price"]
        )

    # Age
    if car_a["vehicle_age"] < car_b["vehicle_age"]:
        newer_name = f"{car_a['brand']} {car_a['model']}"
        age_difference = (
            car_b["vehicle_age"] - car_a["vehicle_age"]
        )
    else:
        newer_name = f"{car_b['brand']} {car_b['model']}"
        age_difference = (
            car_a["vehicle_age"] - car_b["vehicle_age"]
        )

    # Mileage
    if car_a["mileage"] > car_b["mileage"]:
        better_mileage_name = (
            f"{car_a['brand']} {car_a['model']}"
        )
        mileage_difference = (
            car_a["mileage"] - car_b["mileage"]
        )
    else:
        better_mileage_name = (
            f"{car_b['brand']} {car_b['model']}"
        )
        mileage_difference = (
            car_b["mileage"] - car_a["mileage"]
        )

    # Power
    if car_a["max_power"] > car_b["max_power"]:
        powerful_name = (
            f"{car_a['brand']} {car_a['model']}"
        )
        power_difference = (
            car_a["max_power"] - car_b["max_power"]
        )
    else:
        powerful_name = (
            f"{car_b['brand']} {car_b['model']}"
        )
        power_difference = (
            car_b["max_power"] - car_a["max_power"]
        )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            **💰 Lower Price**

            {cheaper_name} is cheaper by
            **₹{price_difference:,.0f}**.

            **📅 Newer Car**

            {newer_name} is newer by
            **{age_difference:.0f} year(s)**.

            **⛽ Better Mileage**

            {better_mileage_name} has higher mileage by
            **{mileage_difference:.1f}**.
            """
        )

    with col2:

        st.markdown(
            f"""
            **⚡ Higher Power**

            {powerful_name} has higher maximum power by
            **{power_difference:.1f}**.

            **⚙️ Transmission**

            {car_a['brand']} {car_a['model']}:
            **{car_a['transmission_type']}**

            {car_b['brand']} {car_b['model']}:
            **{car_b['transmission_type']}**

            **⛽ Fuel Type**

            {car_a['brand']} {car_a['model']}:
            **{car_a['fuel_type']}**

            {car_b['brand']} {car_b['model']}:
            **{car_b['fuel_type']}**
            """
        )

    st.divider()

    st.info(
        "💡 The comparison highlights differences between the "
        "two cars. A higher price or stronger specification "
        "does not automatically mean a car is better — the "
        "right choice depends on your priorities."
    )