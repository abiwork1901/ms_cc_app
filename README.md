# Credit Card Risk Scoring Application

A full-stack application for credit card risk assessment, consisting of a React frontend, FastAPI ML backend, and a core service.

## Project Structure

```
.
├── frontend/           # React TypeScript frontend
├── ml-backend/        # FastAPI ML service
└── src/              # Core service
```

## Components

### Frontend (React TypeScript)
- Modern React application built with TypeScript
- Credit card application form and risk score display
- Located in `frontend/` directory

### ML Backend (Python FastAPI)
- Machine learning service for risk scoring
- REST API endpoints for credit card assessment
- Model training and serving capabilities
- Located in `ml-backend/` directory

### Core Service
- Core business logic implementation
- Located in `src/` directory

## Prerequisites

- Docker and Docker Compose
- Node.js 16+ (for local development)
- Python 3.8+ (for local development)

## Quick Start with Docker

1. Build the containers:
```bash
docker-compose build
```

2. Start the services:
```bash
docker-compose up
```

The services will be available at:
- Frontend: http://localhost:3000
- ML Backend: http://localhost:8000
- Core Service: http://localhost:8080

## Local Development Setup

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### ML Backend Setup
```bash
cd ml-backend
pip install -r requirements.txt
python train.py  # Train the ML model
python serve.py
```

## API Documentation

### ML Backend API

#### POST /score
Score a credit card application

Request body:
```json
{
  "card_number": "4532015112830366",
  "limit": 5000.0
}
```

Response:
```json
{
  "riskScore": 27.3
}
```

#### GET /health
Health check endpoint

Response:
```json
{
  "status": "healthy"
}
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
