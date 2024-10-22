class CombinedDetector {
    constructor() {
        this.nsfwDetector = new window.NsfwDetector();
        this.ageThreshold = 22;
    }

    async initialize() {
        await Promise.all([
            this.nsfwDetector.loadModel(),
            faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
            faceapi.nets.ageGenderNet.loadFromUri('./models')
        ]);
    }

    async analyzeImage(imageUrl) {
        const img = await this.loadImage(imageUrl);
        const [nsfwResult, ageResult] = await Promise.all([
            this.nsfwDetector.isNsfw(imageUrl),
            this.detectAge(img)
        ]);

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
