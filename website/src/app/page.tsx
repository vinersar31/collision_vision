import Image from "next/image";
import styles from "./page.module.css";

export default function Home() {
  return (
    <main className={styles.main}>
      {/* Hero Section */}
      <section className={styles.hero}>
        <h1 className={`${styles.heroTitle} animate-fade-in`}>
          Learning Instance Segmentation with <span className={styles.heroGradient}>CollisionVision</span>
        </h1>
        <p className={`${styles.heroSubtitle} animate-fade-in`} style={{ animationDelay: '0.2s' }}>
          An educational project exploring computer vision, transfer learning, and the mathematics behind YOLOv8-Seg for detecting vehicle damage.
        </p>
        <div className={`${styles.ctaContainer} animate-fade-in`} style={{ animationDelay: '0.4s' }}>
          <a href="#ai-theory" className={styles.primaryCta}>
            Explore the Theory
          </a>
          <a href="https://github.com/vinersar31/collision_vision" target="_blank" rel="noreferrer" className={styles.secondaryCta}>
            View Source Code
          </a>
        </div>
      </section>

      {/* Educational Goals Grid */}
      <section className="container">
        <h2 className={styles.sectionTitle}>What You'll Learn</h2>
        <div className={styles.featuresGrid}>
          <div className={`${styles.featureCard} glass`}>
            <div className={styles.featureIcon}>📐</div>
            <h3 className={styles.featureTitle}>The Math of Segmentation</h3>
            <p className={styles.featureDesc}>
              Understand how models predict pixel-level masks using matrix operations, sigmoid activations, and bounding box offsets.
            </p>
          </div>
          <div className={`${styles.featureCard} glass`}>
            <div className={styles.featureIcon}>🧠</div>
            <h3 className={styles.featureTitle}>Transfer Learning</h3>
            <p className={styles.featureDesc}>
              Learn how to fine-tune pre-trained weights (YOLOv8) on custom datasets without starting from scratch.
            </p>
          </div>
          <div className={`${styles.featureCard} glass`}>
            <div className={styles.featureIcon}>🛠️</div>
            <h3 className={styles.featureTitle}>Data Pipelines</h3>
            <p className={styles.featureDesc}>
              See how raw polygon annotations (VIA/COCO) are converted into the normalized YOLO format for training.
            </p>
          </div>
        </div>
      </section>

      {/* AI and Math Section */}
      <section id="ai-theory" className={`container ${styles.section}`}>
        <h2 className={styles.sectionTitle}>The AI & Math Behind It</h2>
        <div className={styles.aiSection}>
          <div className={`${styles.aiCard} glass`}>
            <div className={styles.featureIcon}>🔍</div>
            <div className={styles.aiCardContent}>
              <h3>YOLOv8 Architecture</h3>
              <p>
                YOLOv8 operates as a single-stage detector. It processes the entire image through a <strong>CSPDarknet backbone</strong> to extract spatial features. The network uses anchor-free detection, predicting the center of an object directly and outputting bounding box coordinates as distance offsets from the center.
              </p>
            </div>
          </div>
          <div className={`${styles.aiCard} glass`}>
            <div className={styles.featureIcon}>🎭</div>
            <div className={styles.aiCardContent}>
              <h3>Generating Masks</h3>
              <p>
                Instance segmentation in YOLOv8 uses two branches: one predicts <em>mask coefficients</em> (a vector for each detected object), and the other generates high-resolution <em>prototype masks</em>. The final segmentation mask is calculated via a matrix multiplication of the prototype masks and the coefficients, followed by a sigmoid activation function to output pixel probabilities: <code>Mask = Sigmoid(Protos × Coeffs)</code>.
              </p>
            </div>
          </div>
          <div className={`${styles.aiCard} glass`}>
            <div className={styles.featureIcon}>📉</div>
            <div className={styles.aiCardContent}>
              <h3>Loss Functions</h3>
              <p>
                Training optimizes a multi-task loss function: 
                <br/>• <strong>Classification Loss (VFL/BCE)</strong>: Penalizes incorrect class predictions.
                <br/>• <strong>Bounding Box Loss (CIoU + DFL)</strong>: Ensures tight fitting boxes.
                <br/>• <strong>Mask Loss (BCE)</strong>: Calculates the binary cross-entropy between the predicted pixel probabilities and the ground truth polygon masks.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Fine Tuning Section */}
      <section className={`container ${styles.section}`}>
        <h2 className={styles.sectionTitle}>Fine-Tuning Process</h2>
        <div className={`${styles.aiCard} glass`} style={{width: '100%'}}>
           <div className={styles.aiCardContent}>
              <p>
                Instead of training a model from random initialization, we use <strong>Fine-Tuning (Transfer Learning)</strong>. We load a model (`yolov8s-seg.pt`) originally trained on the massive COCO dataset (everyday objects) and continue training on our car damage dataset. 
              </p>
              <br/>
              <p>
                During fine-tuning, we leverage the pre-learned feature extractors (edge and texture detectors) in the early layers, while updating the final layers (the detection head) to recognize our specific damage classes like dents and scratches. This drastically reduces the required training data and time while improving accuracy.
              </p>
           </div>
        </div>
      </section>

      {/* Examples Section */}
      <section id="examples" className={`container ${styles.section}`}>
        <h2 className={styles.sectionTitle}>Model Inference</h2>
        <div className={styles.examplesGrid}>
          <div className={`${styles.exampleImageWrapper} glass`}>
            <Image 
              src="/collision_vision/images/examples/11.jpg" 
              alt="Car damage example 1" 
              fill
              loading="eager"
            />
            <div className={styles.exampleLabel}>Raw Input</div>
          </div>
          <div className={`${styles.exampleImageWrapper} glass`}>
            <Image 
              src="/collision_vision/images/examples/12.jpg" 
              alt="Car damage example 2" 
              fill
              loading="eager"
            />
            <div className={styles.exampleLabel}>Segmented Output (Simulated)</div>
          </div>
        </div>
      </section>

      {/* Code Section */}
      <section className={`container ${styles.section}`}>
        <h2 className={styles.sectionTitle}>Running Inference</h2>
        <div className={`${styles.codeBlock} glass`}>
          <div><span className={styles.codeCommand}>import</span> cv2</div>
          <div><span className={styles.codeCommand}>from</span> ultralytics <span className={styles.codeCommand}>import</span> YOLO</div>
          <br/>
          <div className={styles.codeComment}># Load the fine-tuned model weights</div>
          <div>model = YOLO(<span className={styles.codeArg}>"runs/segment/train/weights/best.pt"</span>)</div>
          <br/>
          <div className={styles.codeComment}># Run prediction with a confidence threshold</div>
          <div>results = model.predict(source=<span className={styles.codeArg}>"crash.jpg"</span>, conf=0.25)</div>
          <br/>
          <div className={styles.codeComment}># The results object contains the math output: boxes, masks, and probs</div>
          <div><span className={styles.codeCommand}>for</span> r <span className={styles.codeCommand}>in</span> results:</div>
          <div>&nbsp;&nbsp;&nbsp;&nbsp;masks = r.masks.data <span className={styles.codeComment}># Tensor of shape (N, H, W)</span></div>
          <div>&nbsp;&nbsp;&nbsp;&nbsp;boxes = r.boxes.data <span className={styles.codeComment}># Tensor of shape (N, 6)</span></div>
        </div>
      </section>

      <footer className={styles.footer}>
        <p>Built with Next.js • CollisionVision Educational Project</p>
      </footer>
    </main>
  );
}
