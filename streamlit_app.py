import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment

def main():
    # Set the title of the Streamlit app
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    # Commented out images for a high-level overview and pipeline diagram
    # high_level_image = Image.open("_assets/high_level_overview.png")
    # st.image(high_level_image, caption="High Level Pipeline")

    # Commented out diagram for the full pipeline
    # whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")
    # st.image(whole_pipeline_image, caption="Whole Pipeline")

    # Provide a description of the problem statement
    st.markdown(
        """ 
    #### Problem Statement 
    The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.
    """
    )
    
    # Description of input features and their roles
    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )
    
    # Sidebar inputs for the user to enter feature values
    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.number_input("Payment Value")
    price = st.number_input("Price")
    freight_value = st.number_input("Freight Value")
    product_name_length = st.number_input("Product Name Length")
    product_description_length = st.number_input("Product Description Length")
    product_photos_qty = st.number_input("Product Photos Quantity")
    product_weight_g = st.number_input("Product Weight (grams)")
    product_length_cm = st.number_input("Product Length (CMs)")
    product_height_cm = st.number_input("Product Height (CMs)")
    product_width_cm = st.number_input("Product Width (CMs)")

    # Button for triggering prediction
    if st.button("Predict"):
        # Load prediction service from the deployed pipeline
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )

        # If no service is found, trigger the deployment pipeline to create the service
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_deployment()

        # Prepare the input data in the form of a pandas DataFrame
        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],  # Note: there's a typo in the feature name 'product_name_lenght'
                "product_description_lenght": [product_description_length],  # Typo in 'lenght' should be 'length'
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )

        # Convert the DataFrame to a list of dictionaries, then to a numpy array
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)

        # Make prediction using the service
        pred = service.predict(data)

        # Display the predicted customer satisfaction score
        st.success(
            "Your Customer Satisfaction rate (range between 0 - 5) with the given product details is: {}".format(
                pred
            )
        )
    
    # You can enable this section for additional results analysis (commented out for now)
    # if st.button("Results"):
    #     st.write(
    #         "We have experimented with two ensemble and tree-based models and compared the performance of each model. The results are as follows:"
    #     )

    #     df = pd.DataFrame(
    #         {
    #             "Models": ["LightGBM", "Xgboost"],
    #             "MSE": [1.804, 1.781],
    #             "RMSE": [1.343, 1.335],
    #         }
    #     )
    #     st.dataframe(df)

    #     st.write(
    #         "Following figure shows how important each feature is in the model that contributes to the target variable or contributes to predicting customer satisfaction rate."
    #     )
    #     image = Image.open("_assets/feature_importance_gain.png")
    #     st.image(image, caption="Feature Importance Gain")

if __name__ == "__main__":
    main()
