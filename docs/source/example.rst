Examples
========

ECG MIT-BIH dataset
-------------------

.. toctree::
    :maxdepth: 1

    ./nb/nb_ECG.ipynb

Results
*******

- The localized regions of ECG complexes in sinus rhythm are most informative in distinguishing presence of ventricular ectopic beats from supraventricular ectopic beats in a particular individual. The localized regions lie in the **QRS complex**, which correlates with ventricular depolarization or electrical propagation in the ventricles. Ion channel aberrations and structural abnormalities in the ventricles can affect electrical conduction in the ventricles, manifesting with subtle anomalies in the QRS complex in sinus rhythm that **may not be discernible by the naked eye but is detectable by the convolutional auto-encoder**. Of note, as the $R^2$ increases from 10\% to 88\%, the highlighted color bar is progressively broader, covering a higher proportion of the QRS complex. The foregoing observations are sensible: the regions of interest resided in the QRS complex are biologically plausible and **consistent with cardiac electrophysiological principles**.

- As the R2 increases from 80% to 84% and finally 88%, the blue bar progressively highlights the P wave of ECG complexes in sinus rhythm. This observation is **consistent with our understanding of the mechanistic underpinnings of atrial depolarization**, which correlates with the P wave. Ion channel alterations and structural changes in the atria can affect electrical conduction in the atria manifesting with subtle anomalies in the P wave in sinus rhythm that may not be discernible by the naked eye but are detectable by the convolutional auto-encoder.

- Collectively, the examples underscore the fact that the discriminative regions of interest identified by our proposed method are biologically plausible and consistent with cardiac electrophysiological principles while **locating subtle anomalies in the P wave and QRS complex that may not be discernible by the naked eye**. By inspecting our results with an ECG clinician `Dr. Lin Yee Chen <https://med.umn.edu/bio/cardiovascular/lin-yee>`_, the localized discriminative features of the ECG are consistent with medical interpretation in ECG diagnosis.

MNIST dataset
-------------

.. toctree::
	:maxdepth: 1

	./nb/demo_MNIST.ipynb

