# GREedy Approximation Taylor Selection (GREATS)
> TLDR; Uses Validation Set to select the best possible batch in that iteration...!?
* ![image](https://github.com/user-attachments/assets/8726b27a-bfc1-4e86-85e5-aaea35cf22ed)
* At each training iteration, these methods leverage the partially trained model to determine
which data to select for the current training iteration from a sampled batch, thereby adapting to the
model’s learning progress and focusing on the most informative examples for the model’s current
state
* By continuously updating the selection criteria based on the model’s
progress, online batch selection can identify the most relevant and informative examples at each stage
of training, potentially leading to faster convergence and better generalization performance. Moreover,
online batch selection operates on smaller batches of data, reducing the need for cumbersome data
preprocessing and enabling more efficient use of computational resources compared to static data
selection methods that process the entire dataset upfront
* ![image](https://github.com/user-attachments/assets/861461eb-8747-4ba4-88f8-4ee2cb266a54)
* ![image](https://github.com/user-attachments/assets/c6cf8a25-4c31-4433-b721-95e7a9ddfa3d)
* ![image](https://github.com/user-attachments/assets/32cee938-b530-4628-af24-f761adb6158c)
* ![image](https://github.com/user-attachments/assets/a605f706-2deb-403f-a8cc-5f42feb0bce3)
  - $\mathcal{B}_t$ is the **full set of training data** at time $t$.
  - $\hat{\mathcal{B}}_t$ is the **subset of data points that have already been selected** (e.g., for training or importance sampling).
  - $z*$ belongs to all elements in $\mathcal{B}_t$ but not in $\hat{\mathcal{B}}_t$ because $\hat{\mathcal{B}}_t$ is the selected datapoints, we don't want to use it again for selection
* ![image](https://github.com/user-attachments/assets/4b032411-a6dd-46d9-8c39-cf9d4bd2eb94)
* ![image](https://github.com/user-attachments/assets/a8d72c82-e120-47db-b338-07faba4238d1)
* ![image](https://github.com/user-attachments/assets/0cad3a43-87c0-4b69-8753-4b7d9d959cd8)
* ![image](https://github.com/user-attachments/assets/0cf2235f-62b4-4c01-b977-43fe6472cf2c)
* ![image](https://github.com/user-attachments/assets/65572f7a-2528-44c2-9ed6-3aea3071392f)


# Dataset-Reconstruction
* ![image](https://github.com/user-attachments/assets/1cc6ace8-c206-43a5-8df1-3b73ef4be380)
* ![image](https://github.com/user-attachments/assets/e9190408-0356-421a-928d-a80c4513d3e5)
* ![image](https://github.com/user-attachments/assets/474e971a-e24c-41b5-a4ce-36e84722142d)
* ![image](https://github.com/user-attachments/assets/f72657a2-bfe2-45a0-8d74-a0c499fb11de)
* ![image](https://github.com/user-attachments/assets/7d086cba-6cf0-4d4a-8946-75967d435cd3)
  but it's not an "equal to" there so,
  ![image](https://github.com/user-attachments/assets/3f5f72b7-17fb-4ffd-a451-1c6b70f79adb)
* If the system is consistent, it may have a unique solution or infinitely many solutions.
If it's inconsistent, no exact solution exists, but an approximate solution can be found using least squares.
* ![image](https://github.com/user-attachments/assets/9f7ce299-c927-4462-8fcd-23631df68063)
* ![image](https://github.com/user-attachments/assets/feba4c6d-d2c2-4510-b1ae-cc1b074ac4d2)






