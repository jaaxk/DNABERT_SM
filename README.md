# DNABERT-S with Global Attention Mechanism

With long input sequences, clustering and classification tasks may suffer from a low signal to noise ratio when using mean pooling.
Using weighted-sum pooling with a learning global attention network will help the model understand which token embeddings are more useful for downstream tasks

##What I've implemented
1. Added an attention network in the model's forward method that learns using contrastive loss
2. Moved the model from DataParallel to DistributedDataParallel for more efficient training, and to reduce common errors with DataPatallel
3. Made the model save and load checkpoints so training can be resumed at any point
4. Implemented Fixed-Precision 16 (fp16) for higher memory efficiency
5. Added logging of train and validation loss for analysis

## Training
1. Create python virtual environment with `pip install -r requirements.txt`
2. Run the SLURM script provided `sbatch run_train_ddp.slurm`
