import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Toaster } from "sonner";
import Home from "@/pages/Home";

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-slate-50/50">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/other" element={<div className="text-center text-xl">Other Page - Coming Soon</div>} />
        </Routes>
        <Toaster 
          position="top-right" 
          richColors 
          closeButton
          toastOptions={{
            style: {
              background: 'white',
              border: '1px solid #e2e8f0',
              color: '#0f172a',
            },
          }}
        />
      </div>
    </Router>
  );
}
