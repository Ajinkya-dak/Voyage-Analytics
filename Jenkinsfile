pipeline {
    agent any

    environment {
        VENV_PY = './venv/bin/python3'
        VENV_PIP = './venv/bin/pip3'
        SCRIPT_FILE = 'Flight_price_prediction_model.py'
        IMAGE_NAME = 'flight-price-model'
    }

    stages {
        stage('Set up Python Environment') {
            steps {
                sh '''
                    python3 -m venv venv
                    ./venv/bin/pip install --upgrade pip
                    ./venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Run Model Script') {
            steps {
                sh './venv/bin/python Flight_price_prediction_model.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME -f Dockerfile .'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh 'docker run -d -p 8501:8501 --name flight_model_app $IMAGE_NAME || echo "Container may already be running"'
            }
        }
    }

    post {
        success {
            echo '✅ CI/CD Pipeline completed successfully.'
        }
        failure {
            echo '❌ Pipeline failed. Check logs.'
        }
    }
}
