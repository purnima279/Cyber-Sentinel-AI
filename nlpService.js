const axios = require("axios");

const NLP_SERVICE_URL =
    process.env.NLP_SERVICE_URL || "http://127.0.0.1:8001";


async function analyzeText(text) {
    try {
        const response = await axios.post(
            `${NLP_SERVICE_URL}/analyze`,
            { text }
        );

        console.log("🔎 NLP RESPONSE RECEIVED BY NODE:");
        console.log(JSON.stringify(response.data, null, 2));

        return response.data;
    } catch (error) {
        console.error(
            "NLP Service Error:",
            error.response?.data || error.message
        );
        throw new Error("NLP service is unavailable");
    }
}


async function predictLSTM(text) {
    try {
        const response = await axios.post(
            `${NLP_SERVICE_URL}/nlp/lstm/predict`,
            { text }
        );
        return response.data;
    } catch (error) {
        console.error(
            "LSTM Predict Service Error:",
            error.response?.data || error.message
        );
        throw new Error("LSTM Prediction service failed");
    }
}


async function generateSeq2Seq(text) {
    try {
        const response = await axios.post(
            `${NLP_SERVICE_URL}/nlp/seq2seq/generate`,
            { text }
        );
        return response.data;
    } catch (error) {
        console.error(
            "Seq2Seq Service Error:",
            error.response?.data || error.message
        );
        throw new Error("Seq2Seq Generation service failed");
    }
}


async function getModelMetrics() {
    try {
        const response = await axios.get(
            `${NLP_SERVICE_URL}/nlp/models/metrics`
        );
        return response.data;
    } catch (error) {
        console.error(
            "NLP Model Metrics Error:",
            error.response?.data || error.message
        );
        throw new Error("Failed to fetch NLP model metrics");
    }
}


async function retrainModel() {
    try {
        const response = await axios.post(
            `${NLP_SERVICE_URL}/nlp/train`,
            {}
        );
        return response.data;
    } catch (error) {
        console.error(
            "NLP Retrain Error:",
            error.response?.data || error.message
        );
        throw new Error("Failed to retrain NLP models");
    }
}


module.exports = {
    analyzeText,
    predictLSTM,
    generateSeq2Seq,
    getModelMetrics,
    retrainModel
};