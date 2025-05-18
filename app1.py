import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Green Logistics Optimizer",
    page_icon="🌿",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
    }
    .info-text {
        font-size: 1rem;
    }
    .highlight {
        background-color: #E8F5E9;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.markdown("<h1 class='main-header'>🌿 Green Logistics Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Minimize carbon emissions in your delivery routes using deep learning</p>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Upload & Exploration", "Model Training", "Route Optimization", "Emissions Analysis"])

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scalers' not in st.session_state:
    st.session_state.scalers = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'optimized_routes' not in st.session_state:
    st.session_state.optimized_routes = None

# Function to create and train the deep learning model
def create_emissions_model(X_train, y_train, X_val, y_val):
    # Create a Sequential model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output layer for emissions prediction
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

# Function to preprocess data
def preprocess_data(df):
    # Identify features and target
    X = df.drop(columns=['emissions'])
    y = df['emissions']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

# Function to predict emissions for new routes
def predict_emissions(model, scaler_X, scaler_y, route_data):
    # Scale the input data
    route_data_scaled = scaler_X.transform(route_data)
    
    # Predict emissions
    predicted_emissions_scaled = model.predict(route_data_scaled)
    
    # Inverse transform to get the actual emissions values
    predicted_emissions = scaler_y.inverse_transform(predicted_emissions_scaled.reshape(-1, 1)).flatten()
    
    return predicted_emissions

# Function to optimize routes
def optimize_routes(df, model, scaler_X, scaler_y, selected_features):
    # Get unique routes
    routes = df['route_id'].unique()
    optimized_routes = []
    
    for route in routes:
        route_data = df[df['route_id'] == route].copy()
        
        # Try different combinations of factors that can be controlled
        # For this demo, we'll vary cargo weight within a reasonable range
        original_weight = route_data['cargo_weight'].values[0]
        weights_to_try = np.linspace(max(0.8 * original_weight, 100), min(1.2 * original_weight, 5000), 10)
        
        best_emissions = float('inf')
        best_weight = original_weight
        
        for weight in weights_to_try:
            test_data = route_data.copy()
            test_data['cargo_weight'] = weight
            
            # Prepare data for prediction (extract only the selected features)
            pred_data = test_data[selected_features].copy()
            
            # Predict emissions
            emissions = predict_emissions(model, scaler_X, scaler_y, pred_data)[0]
            
            if emissions < best_emissions:
                best_emissions = emissions
                best_weight = weight
        
        # Store optimized route data
        optimized_route = {
            'route_id': route,
            'original_emissions': route_data['emissions'].values[0],
            'optimized_emissions': best_emissions,
            'emissions_reduction': route_data['emissions'].values[0] - best_emissions,
            'reduction_percentage': (1 - best_emissions / route_data['emissions'].values[0]) * 100,
            'original_weight': original_weight,
            'optimized_weight': best_weight
        }
        optimized_routes.append(optimized_route)
    
    return pd.DataFrame(optimized_routes)

# Home page
if page == "Home":
    st.markdown("<h2 class='sub-header'>Welcome to Green Logistics Optimizer</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight'>
    <p>This application helps logistics companies optimize their delivery routes for minimal carbon emissions using deep learning technology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How it works:")
    st.markdown("""
    1. **Upload your logistics data** containing route information, fuel usage, weather conditions, traffic, and cargo weights
    2. **Explore and visualize** your data to understand emissions patterns
    3. **Train a deep learning model** to predict carbon emissions based on your data
    4. **Optimize routes** to minimize emissions by adjusting controllable factors
    5. **Analyze results** to make informed decisions for greener logistics
    """)
    
    st.markdown("### Key Features:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("- **Data-driven optimization** leveraging your historical delivery data")
        st.markdown("- **Advanced deep learning model** that learns from multiple factors")
        st.markdown("- **Interactive visualizations** to understand emissions patterns")
    
    with col2:
        st.markdown("- **Route-specific recommendations** for immediate implementation")
        st.markdown("- **What-if scenario analysis** to test different strategies")
        st.markdown("- **Emissions reduction metrics** to track environmental impact")
    
    st.info("Navigate through the app using the sidebar to explore all features.")

# Data Upload & Exploration page
elif page == "Data Upload & Exploration":
    st.markdown("<h2 class='sub-header'>Data Upload & Exploration</h2>", unsafe_allow_html=True)
    
    # Option to use sample data or upload own data
    data_option = st.radio("Choose data source", ["Use sample data", "Upload your own data"])
    
    if data_option == "Use sample data":
        # Generate sample data
        st.info("Using synthetic sample data for demonstration purposes.")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        route_ids = [f"R{i:04d}" for i in range(1, 101)] * 10
        distances = np.random.uniform(10, 500, n_samples)  # km
        fuel_usage = distances * np.random.uniform(0.08, 0.15, n_samples)  # liters
        weather_conditions = np.random.choice(['sunny', 'rainy', 'cloudy', 'snowy'], n_samples)
        temperature = np.random.uniform(-5, 35, n_samples)  # Celsius
        traffic_density = np.random.uniform(0.1, 1.0, n_samples)  # scale 0-1
        cargo_weight = np.random.uniform(500, 5000, n_samples)  # kg
        route_type = np.random.choice(['urban', 'highway', 'rural', 'mixed'], n_samples)
        vehicle_type = np.random.choice(['light', 'medium', 'heavy'], n_samples)
        
        # Convert categorical variables to numerical
        weather_mapping = {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'snowy': 3}
        weather_numerical = np.array([weather_mapping[w] for w in weather_conditions])
        
        route_mapping = {'urban': 0, 'rural': 1, 'highway': 2, 'mixed': 3}
        route_numerical = np.array([route_mapping[r] for r in route_type])
        
        vehicle_mapping = {'light': 0, 'medium': 1, 'heavy': 2}
        vehicle_numerical = np.array([vehicle_mapping[v] for v in vehicle_type])
        
        # Calculate emissions (synthetic model)
        base_emissions = fuel_usage * 2.31  # 2.31 kg CO2 per liter of diesel
        weather_factor = 1 + 0.1 * weather_numerical
        traffic_factor = 1 + 0.5 * traffic_density
        weight_factor = 1 + (cargo_weight / 5000) * 0.3
        route_factor = 1 + 0.2 * route_numerical
        vehicle_factor = 1 + 0.3 * vehicle_numerical
        
        emissions = base_emissions * weather_factor * traffic_factor * weight_factor * route_factor * vehicle_factor
        emissions = emissions * np.random.uniform(0.9, 1.1, n_samples)  # Add some randomness
        
        # Create DataFrame
        data = {
            'route_id': route_ids,
            'distance': distances,
            'fuel_usage': fuel_usage,
            'weather_condition': weather_conditions,
            'weather_numerical': weather_numerical,
            'temperature': temperature,
            'traffic_density': traffic_density,
            'cargo_weight': cargo_weight,
            'route_type': route_type,
            'route_numerical': route_numerical,
            'vehicle_type': vehicle_type,
            'vehicle_numerical': vehicle_numerical,
            'emissions': emissions
        }
        
        df = pd.DataFrame(data)
        st.session_state.data = df
    
    else:
        # Upload file
        uploaded_file = st.file_uploader("Upload your logistics data (CSV format)", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Check if the required columns are present
            required_columns = ['route_id', 'distance', 'fuel_usage', 'weather_condition', 
                               'temperature', 'traffic_density', 'cargo_weight', 'emissions']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.markdown("""
                Your data must contain at least the following columns:
                - route_id: Unique identifier for each route
                - distance: Route distance in km
                - fuel_usage: Fuel consumption in liters
                - weather_condition: Weather conditions (e.g., sunny, rainy)
                - temperature: Temperature in Celsius
                - traffic_density: Traffic density (scale 0-1)
                - cargo_weight: Weight of cargo in kg
                - emissions: Carbon emissions in kg CO2
                """)
            else:
                st.success("Data uploaded successfully!")
                
                # Process categorical variables if needed
                if 'weather_numerical' not in df.columns and 'weather_condition' in df.columns:
                    weather_mapping = {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'snowy': 3}
                    df['weather_numerical'] = df['weather_condition'].map(weather_mapping)
                
                if 'route_numerical' not in df.columns and 'route_type' in df.columns:
                    route_mapping = {'urban': 0, 'rural': 1, 'highway': 2, 'mixed': 3}
                    df['route_numerical'] = df['route_type'].map(route_mapping)
                
                if 'vehicle_numerical' not in df.columns and 'vehicle_type' in df.columns:
                    vehicle_mapping = {'light': 0, 'medium': 1, 'heavy': 2}
                    df['vehicle_numerical'] = df['vehicle_type'].map(vehicle_mapping)
                
                st.session_state.data = df
    
    # Data exploration
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.markdown("### Data Overview")
        st.write(df.head())
        
        st.markdown("### Data Statistics")
        st.write(df.describe())
        
        # Data visualizations
        st.markdown("### Data Visualizations")
        
        viz_type = st.selectbox("Choose visualization", 
                               ["Emissions by Route Type", "Emissions vs. Cargo Weight", 
                                "Emissions vs. Distance", "Emissions vs. Weather", 
                                "Correlation Matrix"])
        
        fig = plt.figure(figsize=(10, 6))
        
        if viz_type == "Emissions by Route Type" and 'route_type' in df.columns:
            sns.boxplot(x='route_type', y='emissions', data=df)
            plt.title('Carbon Emissions by Route Type')
            plt.xlabel('Route Type')
            plt.ylabel('Emissions (kg CO2)')
            
        elif viz_type == "Emissions vs. Cargo Weight":
            plt.scatter(df['cargo_weight'], df['emissions'], alpha=0.5)
            plt.title('Carbon Emissions vs. Cargo Weight')
            plt.xlabel('Cargo Weight (kg)')
            plt.ylabel('Emissions (kg CO2)')
            
        elif viz_type == "Emissions vs. Distance":
            plt.scatter(df['distance'], df['emissions'], alpha=0.5)
            plt.title('Carbon Emissions vs. Distance')
            plt.xlabel('Distance (km)')
            plt.ylabel('Emissions (kg CO2)')
            
        elif viz_type == "Emissions vs. Weather" and 'weather_condition' in df.columns:
            sns.boxplot(x='weather_condition', y='emissions', data=df)
            plt.title('Carbon Emissions by Weather Condition')
            plt.xlabel('Weather Condition')
            plt.ylabel('Emissions (kg CO2)')
            
        elif viz_type == "Correlation Matrix":
            # Select only numerical columns
            numerical_df = df.select_dtypes(include=['float64', 'int64'])
            correlation = numerical_df.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix')
            
        st.pyplot(fig)

# Model Training page
elif page == "Model Training":
    st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload or generate data first in the 'Data Upload & Exploration' page.")
    else:
        df = st.session_state.data
        
        st.markdown("### Feature Selection")
        st.write("Select features to include in the model:")
        
        categorical_features = ['weather_numerical', 'route_numerical', 'vehicle_numerical']
        numerical_features = ['distance', 'fuel_usage', 'temperature', 'traffic_density', 'cargo_weight']
        
        # Let user select features
        selected_categorical = []
        for feature in categorical_features:
            if feature in df.columns:
                if st.checkbox(feature, value=True):
                    selected_categorical.append(feature)
        
        selected_numerical = []
        for feature in numerical_features:
            if feature in df.columns:
                if st.checkbox(feature, value=True):
                    selected_numerical.append(feature)
        
        selected_features = selected_categorical + selected_numerical
        
        if len(selected_features) < 2:
            st.warning("Please select at least two features for the model.")
        else:
            st.markdown("### Training Configuration")
            
            # Training parameters
            test_size = st.slider("Test set size (%)", 10, 30, 20) / 100
            
            if st.button("Train Model"):
                with st.spinner("Training model... This may take a few minutes."):
                    # Prepare the training data
                    model_df = df[selected_features + ['emissions']].copy()
                    model_df = model_df.dropna()  # Remove any rows with missing values
                    
                    # Preprocess data
                    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(model_df)
                    
                    # Split validation set
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                    
                    # Train the model
                    model, history = create_emissions_model(X_train, y_train, X_val, y_val)
                    
                    # Evaluate the model
                    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
                    
                    # Save to session state
                    st.session_state.model = model
                    st.session_state.scalers = {'X': scaler_X, 'y': scaler_y}
                    st.session_state.training_history = history
                    st.session_state.selected_features = selected_features
                    
                    st.success("Model trained successfully!")
                    st.write(f"Test Mean Absolute Error: {test_mae:.2f}")
                    
                    # Plot training history
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    ax1.plot(history.history['loss'])
                    ax1.plot(history.history['val_loss'])
                    ax1.set_title('Model Loss')
                    ax1.set_ylabel('Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.legend(['Train', 'Validation'], loc='upper right')
                    
                    ax2.plot(history.history['mae'])
                    ax2.plot(history.history['val_mae'])
                    ax2.set_title('Model MAE')
                    ax2.set_ylabel('MAE')
                    ax2.set_xlabel('Epoch')
                    ax2.legend(['Train', 'Validation'], loc='upper right')
                    
                    st.pyplot(fig)
                    
                    # Feature importance approximation
                    st.markdown("### Feature Importance")
                    st.write("Approximated feature importance based on weights:")
                    
                    # Get the weights from the first layer
                    weights = model.layers[0].get_weights()[0]
                    
                    # Calculate importance as the sum of absolute weights for each feature
                    importance = np.sum(np.abs(weights), axis=1)
                    importance = importance / np.sum(importance)  # Normalize
                    
                    # Create a DataFrame for visualization
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': importance
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                    ax.set_title('Feature Importance')
                    ax.set_xlabel('Importance')
                    ax.set_ylabel('Feature')
                    
                    st.pyplot(fig)
                    
                    # Option to save model
                    if st.button("Save Model"):
                        # Create directory if it doesn't exist
                        if not os.path.exists('models'):
                            os.makedirs('models')
                        
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_path = f"models/emissions_model_{timestamp}.h5"
                        scaler_path = f"models/scalers_{timestamp}.pkl"
                        
                        # Save model and scalers
                        model.save(model_path)
                        joblib.dump(st.session_state.scalers, scaler_path)
                        
                        st.success(f"Model saved successfully to {model_path}")
                        st.info(f"Scalers saved to {scaler_path}")

# Route Optimization page
elif page == "Route Optimization":
    st.markdown("<h2 class='sub-header'>Route Optimization</h2>", unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' page.")
    elif st.session_state.data is None:
        st.warning("Please upload or generate data first in the 'Data Upload & Exploration' page.")
    else:
        st.markdown("### Optimize Routes for Minimal Emissions")
        
        if st.button("Run Optimization"):
            with st.spinner("Optimizing routes..."):
                # Get the data
                df = st.session_state.data
                model = st.session_state.model
                scaler_X = st.session_state.scalers['X']
                scaler_y = st.session_state.scalers['y']
                selected_features = st.session_state.selected_features
                
                # Check if we have the necessary data
                if 'cargo_weight' not in df.columns:
                    st.error("Error: The dataset must contain a 'cargo_weight' column for optimization.")
                else:
                    try:
                        # Run optimization
                        optimized_routes = optimize_routes(df, model, scaler_X, scaler_y, selected_features)
                        
                        # Save to session state
                        st.session_state.optimized_routes = optimized_routes
                        
                        st.success("Routes optimized successfully!")
                        
                        # Display results
                        st.markdown("### Optimization Results")
                        st.write(optimized_routes)
                        
                        # Summary statistics
                        avg_reduction = optimized_routes['reduction_percentage'].mean()
                        total_emissions_saved = optimized_routes['emissions_reduction'].sum()
                        
                        st.markdown(f"**Average emissions reduction: {avg_reduction:.2f}%**")
                        st.markdown(f"**Total emissions saved: {total_emissions_saved:.2f} kg CO2**")
                        
                        # Visualize results
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Bar chart of emissions reduction by route
                        routes_to_plot = optimized_routes.sort_values('reduction_percentage', ascending=False).head(10)
                        sns.barplot(x='route_id', y='reduction_percentage', data=routes_to_plot, ax=ax1)
                        ax1.set_title('Top 10 Routes by Emissions Reduction')
                        ax1.set_xlabel('Route ID')
                        ax1.set_ylabel('Reduction (%)')
                        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
                        
                        # Scatter plot of original vs optimized emissions
                        ax2.scatter(optimized_routes['original_emissions'], optimized_routes['optimized_emissions'], alpha=0.6)
                        ax2.plot([0, optimized_routes['original_emissions'].max()], [0, optimized_routes['original_emissions'].max()], 'r--')
                        ax2.set_title('Original vs. Optimized Emissions')
                        ax2.set_xlabel('Original Emissions (kg CO2)')
                        ax2.set_ylabel('Optimized Emissions (kg CO2)')
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error during optimization: {str(e)}")
                        st.info("Try training the model again with all necessary features included.")

# Emissions Analysis page
elif page == "Emissions Analysis":
    st.markdown("<h2 class='sub-header'>Emissions Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.optimized_routes is None:
        st.warning("Please run route optimization first in the 'Route Optimization' page.")
    else:
        optimized_routes = st.session_state.optimized_routes
        
        st.markdown("### Emissions Reduction Analysis")
        
        # Overall statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_reduction = optimized_routes['reduction_percentage'].mean()
            st.metric("Average Reduction", f"{avg_reduction:.2f}%")
        
        with col2:
            total_saved = optimized_routes['emissions_reduction'].sum()
            st.metric("Total CO2 Saved", f"{total_saved:.2f} kg")
        
        with col3:
            max_reduction = optimized_routes['reduction_percentage'].max()
            st.metric("Max Reduction", f"{max_reduction:.2f}%")
        
        # Visualizations
        st.markdown("### Detailed Analytics")
        
        # Histogram of emission reductions
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(optimized_routes['reduction_percentage'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribution of Emission Reductions')
        ax.set_xlabel('Reduction (%)')
        ax.set_ylabel('Frequency')
        
        st.pyplot(fig)
        
        # Environmental impact
        st.markdown("### Environmental Impact")
        
        trees_equivalent = total_saved / 21  # Approx. 21 kg CO2 absorbed by one tree per year
        
        st.markdown(f"The total emissions reduction of **{total_saved:.2f} kg CO2** is equivalent to:")
        st.markdown(f"- The annual CO2 absorption of approximately **{trees_equivalent:.1f} trees**")
        st.markdown(f"- Removing about **{(total_saved / 4600):.2f} cars** from the road for a year")
        
        # Route details
        st.markdown("### Route-Specific Details")
        
        selected_route = st.selectbox("Select route to analyze", optimized_routes['route_id'].unique())
        
        route_data = optimized_routes[optimized_routes['route_id'] == selected_route].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Route")
            st.markdown(f"**Emissions:** {route_data['original_emissions']:.2f} kg CO2")
            st.markdown(f"**Cargo Weight:** {route_data['original_weight']:.2f} kg")
        
        with col2:
            st.markdown("#### Optimized Route")
            st.markdown(f"**Emissions:** {route_data['optimized_emissions']:.2f} kg CO2")
            st.markdown(f"**Cargo Weight:** {route_data['optimized_weight']:.2f} kg")
            st.markdown(f"**Reduction:** {route_data['reduction_percentage']:.2f}%")
        
        # Interactive analysis
        st.markdown("### What-If Analysis")
        st.markdown("Adjust parameters to see potential emissions impacts:")
        
        # Get original data
        if st.session_state.data is None:
            st.warning("Original data not available for what-if analysis.")
        else:
            df = st.session_state.data
            model = st.session_state.model
            scaler_X = st.session_state.scalers['X']
            scaler_y = st.session_state.scalers['y']
            
            # Get a sample route
            route_ids = df['route_id'].unique()
            selected_route_what_if = st.selectbox("Select route", route_ids)
            
            route_data = df[df['route_id'] == selected_route_what_if].iloc[0].copy()
            
            # Create sliders for adjustable parameters
            col1, col2 = st.columns(2)
            
            with col1:
                if 'cargo_weight' in df.columns:
                    new_weight = st.slider("Cargo Weight (kg)", 
                                          min_value=float(route_data['cargo_weight'] * 0.5),
                                          max_value=float(route_data['cargo_weight'] * 1.5),
                                          value=float(route_data['cargo_weight']))
            
            with col2:
                if 'traffic_density' in df.columns:
                    new_traffic = st.slider("Traffic Density (0-1)", 
                                           min_value=0.0,
                                           max_value=1.0,
                                           value=float(route_data['traffic_density']))
            
            # Update the route data
            modified_route = route_data.copy()
            if 'cargo_weight' in df.columns:
                modified_route['cargo_weight'] = new_weight
            if 'traffic_density' in df.columns:
                modified_route['traffic_density'] = new_traffic
            
            # Prepare data for prediction
            features = st.session_state.selected_features
            
            pred_data = pd.DataFrame([modified_route[features]])
            
            # Predict emissions
            predicted_emissions = predict_emissions(model, scaler_X, scaler_y, pred_data)
            
            # Display results
            st.markdown(f"### Predicted Emissions: {predicted_emissions[0]:.2f} kg CO2")
            
            # Compare with original
            original_emissions = route_data['emissions']
            change = (predicted_emissions[0] - original_emissions) / original_emissions * 100
            
            st.metric("Emissions Change", f"{change:.2f}%", 
                     delta=-change, delta_color="inverse")
            
            # Recommendation
            if change < 0:
                st.success("This configuration reduces emissions! Consider implementing these changes.")
            else:
                st.warning("This configuration increases emissions. Try adjusting parameters further.")
            
            # Export optimization report
            if st.button("Export Optimization Report"):
                report = pd.DataFrame({
                    'Route ID': [route_data['route_id']],
                    'Original Emissions (kg CO2)': [original_emissions],
                    'Optimized Emissions (kg CO2)': [predicted_emissions[0]],
                    'Reduction (%)': [-change],
                    'Original Cargo Weight (kg)': [route_data['cargo_weight']],
                    'Optimized Cargo Weight (kg)': [new_weight]
                })
                
                st.dataframe(report)
                
                # Create download link
                csv = report.to_csv(index=False)
                st.download_button(
                    label="Download Report as CSV",
                    data=csv,
                    file_name=f"route_optimization_{selected_route_what_if}.csv",
                    mime="text/csv"
                )

# Add a footer
st.markdown("---")
st.markdown("<center>© 2025 Green Logistics Optimizer | Powered by Deep Learning</center>", unsafe_allow_html=True)