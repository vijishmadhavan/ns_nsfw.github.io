class NsfwClassifier {
    constructor() {
        this.model = null;
        this.MODEL_URL = './models/mobilenet_v2/model.json';
        this.classes = ['drawing', 'hentai', 'neutral', 'porn', 'sexy'];
    }

    async loadModel() {
        if (!this.model) {
            console.log('Loading model from:', this.MODEL_URL);
            this.model = await tf.loadLayersModel(this.MODEL_URL);
            console.log('Model loaded:', this.model);
        }
        return this.model;
    }

    async classifyImage(imageElement) {
        const model = await this.loadModel();
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeBilinear([224, 224]) // MobileNet V2 expects 224x224 images
            .toFloat()
            .div(tf.scalar(255))
            .expandDims();
        const predictions = await model.predict(tensor);
        return predictions.dataSync();
    }

    async isNsfw(imageUrl) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = imageUrl;

        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
        });

        const predictions = await this.classifyImage(img);
        const results = this.classes.map((className, index) => ({
            className,
            probability: predictions[index]
        }));

        // Sort results by probability in descending order
        results.sort((a, b) => b.probability - a.probability);

        // Determine if the image is NSFW
        const isNSFW = predictions[1] > 0.2 || predictions[3] > 0.2 || predictions[4] > 0.2;

        const result = {
            isNSFW,
            results
        };
        console.log('NSFW detection result:', result);
        return result;
    }
}

class NsfwDetector {
    constructor() {
        this.nsfwClassifier = new NsfwClassifier();
        this.ageThreshold = 22;
        this.neutralThreshold = 0.9; // 90% threshold for neutral class
    }

    async initialize() {
        await Promise.all([
            this.nsfwClassifier.loadModel(),
            faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
            faceapi.nets.ageGenderNet.loadFromUri('./models')
        ]);
    }

    async analyzeImage(imageUrl) {
        const nsfwResult = await this.nsfwClassifier.isNsfw(imageUrl);

        if (nsfwResult.isNSFW) {
            // If already NSFW, skip age detection
            return {
                isNSFW: true,
                age: null,
                nsfwResults: nsfwResult.results
            };
        }

        // Check if the neutral class probability is above the threshold
        const neutralProb = nsfwResult.results.find(r => r.className === 'neutral')?.probability || 0;
        if (neutralProb > this.neutralThreshold) {
            // If highly neutral, skip age detection
            return {
                isNSFW: false,
                age: null,
                nsfwResults: nsfwResult.results
            };
        }

        // Only load the image and perform age detection if not NSFW and not highly neutral
        const img = await this.loadImage(imageUrl);
        const ageResult = await this.detectAge(img);

        const isNSFW = nsfwResult.isNSFW || (ageResult !== null && ageResult < this.ageThreshold);

        return {
            isNSFW,
            age: ageResult,
            nsfwResults: nsfwResult.results
        };
    }

    async isNsfw(imageUrl) {
        const result = await this.analyzeImage(imageUrl);
        return {
            isNSFW: result.isNSFW,
            results: result.nsfwResults
        };
    }

    async loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    }

    async detectAge(img) {
        const detections = await faceapi
            .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
            .withAgeAndGender();

        if (detections.length > 0) {
            return Math.round(detections[0].age);
        }
        return null;
    }
}

// Make both classes available globally
window.NsfwClassifier = NsfwClassifier;
window.NsfwDetector = NsfwDetector;


