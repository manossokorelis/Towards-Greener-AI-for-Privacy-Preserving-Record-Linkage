# Towards-Greener-AI-for-Privacy-Preserving-Record-Linkage
My bachelor thesis analyses a novel entity resolution process which involves Bloom filters, differential privacy, and deep learning. The work specifically focuses on examining the balance between privacy, accuracy, and energy efficiency within this process. Initially, it was explored at how different configurations of Bloom filters, neural network models and training parameters affect privacy, accuracy and energy consumption. The goal was to select parameters which ensure high performance while minimizing energy usage. In addition, the research focused on reducing the energy consumption of the ER process by experimenting with few-shot learning techniques and cross-dataset evaluation methods. The results showed that the best configuration depends on what you prioritise. For example, small Bloom filters are more energy efficient than large ones but give less accurate results. Furthermore, there is a trade off between privacy and accuracy: adding more noise to the filters makes the data more secure but at the cost of lower accuracy. Finally, experimenting with smaller training sets and model reuse strategies we showed that we can reduce energy consumption but not without compromising model accuracy.

Key Achievements:
- Designed and implemented an innovative ER process that integrates differential privacy and deep learning with Bloom filters to enhance data privacy while maintaining high model performance.
- Conducted a comprehensive analysis of how different configurations of Bloom filters, neural network models, and training parameters impact privacy preservation, accuracy, and energy consumption. Highlighted the intricate balance required for optimal performance.
- Focused on reducing the energy consumption of deep learning models during both training and inference phases by experimenting with smaller training sets (few-shot learning) and cross-dataset evaluation techniques, ensuring that energy usage was minimized without significant loss in accuracy or privacy.
- Utilized tools like PyRAPL to measure CPU energy consumption and RAM usage in real-time, offering novel insights into the computational cost of privacy-preserving methods integrated into ER workflows.
- Explored cross-dataset evaluation and model reuse strategies to show how pre-trained models could save energy and reduce training time, while ensuring model performance remained consistent across different datasets.

