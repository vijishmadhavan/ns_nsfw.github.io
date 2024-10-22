class CombinedDetector {
    constructor() {
        this.nsfwDetector = new window.NsfwDetector();
        this.ageThreshold = 22;
        this.neutralThreshold = 0.9; // 90% threshold for neutral class
    }

    async initialize() {
        await Promise.all([
            this.nsfwDetector.loadModel(),
            faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
            faceapi.nets.ageGenderNet.loadFromUri('./models')
        ]);
    }

    async analyzeImage(imageUrl) {
        const nsfwResult = await this.nsfwDetector.isNsfw(imageUrl);

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

window.CombinedDetector = CombinedDetector;

