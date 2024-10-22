class NsfwDetector {
    constructor() {
        this.model = null;
        this.MODEL_URL = './models/inception_v3/model.json';
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
            .resizeBilinear([299, 299])
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

window.NsfwDetector = NsfwDetector;

}

window.NsfwDetector = NsfwDetector;
