# MLOps Project (DevOps for ML)

### Acknowledgements
---
I would like to extend my sincere thanks to [freeCodeCamp](https://www.freecodecamp.org/) and [Ayush Singh](https://www.youtube.com/@AyushSinghSh) for their invaluable content and guidance, which helped me build this project. This project wouldn't have been possible without their educational resources. I am also grateful to [Hillary Nyakundi](https://www.freecodecamp.org/news/author/larymak/) for providing an excellent resource ([LINK](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)) on how to write a well-structured README.md file.

<br>
<br>

### Introduction to MLOps
---
**MLOps**, short for [Machine Learning Operations](https://aws.amazon.com/what-is/mlops/#:~:text=Machine%20learning%20operations%20(MLOps)%20are,deliver%20value%20to%20your%20customers.), brings the practices of DevOps to the world of machine learning, helping teams build, deploy, and maintain ML models more effectively. By automating repetitive tasks like testing, deployment, and monitoring, MLOps allows data scientists and engineers to focus on improving model accuracy and performance. It makes scaling models easier, keeps them reliable over time, and ensures that they stay relevant by catching issues early—like when a model's accuracy drops or when it starts to drift from real-world data. MLOps bridges the gap between development and production, making sure that the hard work of building models actually makes it into the hands of users smoothly and effectively.

<br>
<br>

### About the Project
---
This project, **_Customer Satisfaction Analysis based on Purchased Goods History_**, explores customer satisfaction by analyzing purchase history and leverages MLOps practices to make it ready for production deployment. While the problem of predicting customer satisfaction was the core focus, the project mainly served as a hands-on application of MLOps concepts using **ZenML**, **MLflow**, and **Streamlit**.

The project used these tools as follows 🔻

1.  **[ZenML](https://www.zenml.io/)** ▶️ Helped design and orchestrate modular machine learning pipelines, providing a structured, repeatable workflow that took us from data processing to deployment.
2.  **[MLflow](https://mlflow.org/)** ▶️ Integrated for model tracking, versioning, and experiment management, making it easy to log metrics, parameters, and model versions to improve and iterate on the model.
3.  **[Streamlit](https://streamlit.io/)** ▶️ Used to create a user-friendly web app interface for stakeholders, allowing non-technical users to interact with the model's predictions, making the customer satisfaction insights accessible and actionable.

These tools worked together to create an efficient, scalable MLOps framework that highlighted the importance of automation, monitoring, and continuous improvement in machine learning projects.

<br>
<br>

### Motivation
----------
The motivation behind this project was to gain hands-on experience with **MLOps** principles, using a practical application as the foundation. By learning how to streamline workflows and automate tasks like testing, deployment, and monitoring, I aimed to understand how machine learning models can be made more robust, scalable, and production-ready. This project was a step towards mastering the tools and processes needed to effectively deploy ML models in real-world environments.

<br>
<br>

### Why I Worked on This Project ?

----------

I worked on this project to deepen my understanding of **MLOps workflows** and the tools that support them, like ZenML, MLflow, and Streamlit. Through this project, I followed a structured, step-by-step approach to developing a machine learning pipeline, guided by Ayush Singh’s tutorial on freeCodeCamp's YouTube channel. The project gave me practical exposure to building a pipeline from scratch, making it easier to understand and implement MLOps in future projects. The **primary goal** was not just the model itself but mastering the end-to-end workflow, from development through to production deployment.

<br>
<br>

### Problem It Solves
----------
The project tackles two areas🔻

1.  **Customer Insight** ▶️ By predicting customer satisfaction from purchase history, the model provides insights into customer preferences and needs.
2.  **Operationalization of ML Models** ▶️ Beyond just creating a model, this project demonstrates how to deploy, monitor, and manage a machine learning solution at scale. This end-to-end approach ensures the model remains reliable, relevant, and efficient over time, making it ready for deployment in a production-like environment.

<br>
<br>

### What I Learned ?
----------
This project was a practical learning experience in applying MLOps to a real-world dataset. Key takeaways included🔻

-   **Pipeline Automation** ▶️ With ZenML, I understood the importance of modular, repeatable workflows that automate data ingestion, model training, and evaluation.
-   **Model Tracking and Experimentation** ▶️ Using MLflow, I learned the value of version control for models, as well as tracking metrics and parameters to continuously improve the model.
-   **UI Development with Streamlit** ▶️ Creating a Streamlit app taught me how to make machine learning insights accessible and user-friendly, providing an intuitive interface for non-technical stakeholders.
-   **Production Challenges and Solutions** ▶️ Through implementing MLOps practices, I learned about the complexities of deploying, monitoring, and maintaining ML models in production, as well as how to address these challenges with robust, automated solutions.

<br>
<br>

### What Makes This Project Stand Out ?
----------
This project is unique because it goes beyond traditional machine learning development, applying MLOps principles to create a fully operationalized solution. The project’s structured pipeline and interactive interface make it a holistic demonstration of how to take a model from a development environment to a scalable production-ready system. The use of ZenML, MLflow, and Streamlit highlights how these tools can work together to enhance automation, versioning, and accessibility, ensuring that the model delivers consistent and reliable insights in a user-friendly way.

<br>
<br>


### Challenge(s) I Faced
---
<b>Tool Integration ▶️ </b>Configuring and integrating ZenML, MLflow, and Streamlit into a smooth workflow posed initial setup challenges.

<br>
<br>

### Features I Hope to Implement in the Future:
---
1.  **Automated Retraining ▶️** Implement continuous retraining pipelines to keep models up to date with new customer data.
2.  **Real-Time Updates ▶️** Enable real-time prediction updates as new data comes in, ensuring dynamic customer satisfaction scores.
3.  **Model Interpretability ▶️** Integrate tools to explain model predictions, enhancing trust and transparency for stakeholders.
4.  **Cross-Platform Deployment ▶️** Ensure smooth deployment on multiple platforms, especially Windows, by addressing daemon functionality limitations.
5. **Deployment on Windows ▶️** MLflow's daemon functionality doesn’t support Windows, requiring adjustments or consideration of alternative environments for deployment.

<br>
<br>

### Running Locally
---
⚠️ **CAUTION ▶️** Follow the steps in order, or you might end up with 🤯💥!
<br>

 - <b>Cloning the Project Directory : </b>`git clone git@github.com:SoubhikSinha/MLOps-Project-DevOps-for-ML.git`

 - <b>Creating Virtual Environment : </b> This project is made on Python 3.10 and the virtual environment for the project was made using Anaconda🔻<br>
	 >  `conda create --prefix ./customer-satisfaction python=3.10`
 
	 <b>🌟 NOTE : </b> If Anaconda, in your local system, has been declared as "Global" (i.e. putting Anaconda in System Variables Path), then you can directly access your `conda` environment from Terminal (else, Anaconda Prompt will be at rescue).

 - <b>Installing required libraries : </b>`pip install -r requirements.txt`

 - <b>Installing the ZenML Server : </b>`pip install "zenml[server]"`<br>
  🌟<b>NOTE : </b> Please ensure that the ZenML Server version and the ZenML Library version are same. If you want to work on a specific version, you shall make use of this command : `pip install "zenml[server]==<version>"` (e.g. >> `pip install "zenml[server]==0.68.1"`)

 - <b>Initialize ZenML repository : </b>`zenml init`

 - <b>Starting the Zenml Server : </b>`zenml up`<br>
 🌟 <b>NOTE : </b>🔻
	 1. If the above doesn't work (especially if you are using a Windows Machine), you shall make use of `zenml up --blocking` OR `zenml up --docker` (Only for the 2<sup>nd</sup> option : Make sure that you have Docker Desktop installed and running).
	 2.  Once you create the local server, you will be able to see an IP address to the ZenML Dashboard (e.g. :  `http://127.0.0.1:8237`).
	 3. The Dashboard requires credentials to enter🔻
		 - Username : **default**
		 - Password : Keep it empty / blank
	4. To kill the server, Press `Ctrl+C` in the terminal / CLI. (Alternatively you may use : `zenml down`)
	5. To remove the connection between your local repository and the remote metadata storage or orchestration setup you are using. : `zenml disconnect`

 - <b>Install the ZenML integration for MLflow : </b>`zenml integration install mlflow -y`

 - <b>Register a new MLflow experiment tracker in ZenML : </b>`zenml experiment-tracker register mlflow_tracker_customer --flavor=mlflow`

 - <b>Register a new MLflow model deployer in ZenML : </b>`zenml model-deployer register mlflow_customer --flavor=mlflow`

- <b>Register a new ZenML stack, associating it with the default orchestrator, artifact store , and experiment tracker : </b>`zenml stack register mlflow_stack_customer -a default -o default -d mlflow_customer -e mlflow_tracker_customer --set`

 - <b>List of all the ZenML stacks that are opened : </b>`zenml stack list`

 - <b>To obtain detailed information about the active stack in your ZenML setup : </b> `zenml stack describe`

 - <b>Running the Pipeline : </b>`python run_pipeline.py`

 - <b>Launch the MLflow UI, connecting it to a specific local backend store to view and manage MLflow experiment logs :</b>`mlflow ui --backend-store-uri file:C:\Users\<YOUR_USERNAME>\AppData\Roaming\zenml\local_stores\4c2e4651-b6c6-48d0-b5aa-e5488be6f979\mlruns` (You can find this file path after you run the `python run_pipeline.py` command).<br>
 🌟 <b>NOTE :</b> After you try to launch the MLFlow Dashboard using the above mentioned command, you will be provided with an IP Address (e.g. :  `http://127.0.0.1:5000`) to go to it. There you shall be able to navigate and check the scores of various metrics considered for the model performance.

-  <b>Locally Deploy the model on "<i>Deployment</i>" Configuration Settings : </b>`python run_deployment.py --config deploy`

- <b>Locally Deploy the model on "<i>Prediction</i>" Configuration Settings : </b>`python run_deployment.py --config predict`

- <b>Run the Streamlit Application : </b>`streamlit run streamlit_app.py`

<br>
<br>

### Conclusion
---
This project has been an exciting journey through MLOps principles and practices, where I’ve gained hands-on experience using tools like ZenML, MLflow, and Streamlit to build a customer satisfaction analysis pipeline. The process has not only deepened my understanding of machine learning workflows but also highlighted the challenges and rewards of deploying machine learning models in real-world environments.

I hope this project provides valuable insights into how MLOps can streamline model development, deployment, and monitoring, and I look forward to continuing to improve and expand this work.
