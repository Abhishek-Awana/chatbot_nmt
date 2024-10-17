import "./App.css";
import LLMInterface from "./Components/LLMInterface.tsx";
import ParticlesBackground from "./Components/Particles.jsx";

function App() {
  return (
    <div className="relative h-screen flex items-center justify-center">
      {/* Particles in the background */}
      <div className="absolute inset-0 z-0">
        <ParticlesBackground />
      </div>

      {/* Main content, overlaid on top of particles */}
      <div className="relative z-10 w-full h-[10rem] flex flex-col-reverse justify-center">
        <LLMInterface />
      </div>
    </div>
  );
}

export default App;