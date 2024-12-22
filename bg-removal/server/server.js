import cors from 'cors';
import 'dotenv/config';
import express from 'express';
import connectDB from './configs/mongodb.js';
import imageRouter from './routes/imageRoutes.js';
import userRouter from './routes/userRoutes.js';

const PORT = process.env.PORT || 4000
const app = express();
await connectDB()

app.use(express.json())
app.use(cors())

// API routes
app.use('/api/user',userRouter)
app.use('/api/image',imageRouter)

app.get('/', (req,res) => res.send("API Working"))

app.listen(PORT, () => console.log('Server running on port ' + PORT));
