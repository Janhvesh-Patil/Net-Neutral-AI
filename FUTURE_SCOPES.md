# Net-Neutral AI: Future Scopes

While the current prototype successfully demonstrates Federated Learning, there are several key architectural and user experience improvements planned for future iterations.

## 1. Local Coordinator Mode (Bypass Cloud Limitations)
**The Problem:** Currently, the Coordinator backend runs entirely on Render. Render's free tier has a strict 512MB RAM limit. While we optimized the evaluation step to prevent crashes, running Federated Averaging (FedAvg) and model evaluation on a massive global model will inevitably strain cloud resources. 
**The Solution:** 
- Shift the Coordinator backend to run **locally** on the Coordinator's own high-performance device.
- Render (or a similar cloud service) would be downgraded to a lightweight signaling relay, only responsible for HTTP routing and tracking IP addresses.
- **Implementation:** Create a "Coordinator Agent Setup Guide" (similar to the current Client Guide) that provides CLI commands for the Coordinator to launch the heavy aggregation and evaluation server on their local machine, ensuring virtually unlimited memory and compute power.

## 2. Frontend & UX Overhaul
**The Problem:** The current user interface is highly basic. Furthermore, there is a massive disparity between the rich data logged in the local command prompt and the sparse data displayed on the web dashboard.
**The Solution:**
- **Real-Time Data Streaming:** Update the dashboard to use WebSockets or Server-Sent Events (SSE) so that training issues, round progression, and status changes update instantly without requiring page refreshes or sluggish polling.
- **Client Dashboard Parity:** Surface the rich terminal data in the UI. The Client Dashboard should display live per-epoch loss, local accuracy, credit calculations, and training duration graphs.
- **Aesthetic Redesign:** Implement a modern, visually appealing UI framework (e.g., React/Tailwind) to replace the vanilla prototype styling, making it presentation-ready.

## 3. Model Export & Deployment
**The Problem:** After a 5-round training session completes, the final trained Global Model (`checkpoint.pt`) sits on the Coordinator's server, but there is no easy way for the user to retrieve it.
**The Solution:**
- Add a **"Download Global Model"** button to the Coordinator Dashboard that appears once training is finalized.
- Allow the Coordinator to export the PyTorch weights locally so they can use the collaboratively trained model in external applications or production environments.
