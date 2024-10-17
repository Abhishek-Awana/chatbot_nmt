import { useEffect, useMemo, useState } from "react";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim"; // Loading only the necessary features

const ParticlesBackground = () => {
  const [init, setInit] = useState(false);

  // Initialize the particles engine once
  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine); // load slim version for better performance
    }).then(() => {
      setInit(true);
    });
  }, []);

  // Memoized particles options for better performance
  const options = useMemo(
    () => ({
      background: {
        color: {
          value: "#ff00ff", // Fallback magenta color
        },
        image: "linear-gradient(135deg, #ff00ff 0%, #ff77ff 100%)", // Magenta gradient
      },
      fpsLimit: 120,
      interactivity: {
        events: {
          onClick: {
            enable: true,
            mode: "push",
          },
          onHover: {
            enable: true,
            mode: "repulse",
          },
        },
        modes: {
          push: {
            quantity: 4,
          },
          repulse: {
            distance: 200,
            duration: 0.4,
          },
        },
      },
      particles: {
        color: {
          value: "#fed7aa", // White particles
        },
        links: {
          color: "#ffffff",
          distance: 150,
          enable: true,
          opacity: 0.5,
          width: 1,
        },
        move: {
          direction: "none",
          enable: true,
          outModes: {
            default: "bounce",
          },
          random: false,
          speed: 6,
          straight: false,
        },
        number: {
          density: {
            enable: true,
          },
          value: 450,
        },
        opacity: {
          value: 0.5,
        },
        shape: {
          type: "circle",
        },
        size: {
          value: { min: 1, max: 5 },
        },
      },
      detectRetina: true,
    }),
    [],
  );

  if (init) {
    return (
      <Particles
        id="tsparticles"
        options={options}
        particlesLoaded={(container) => console.log(container)}
      />
    );
  }

  return null;
};

export default ParticlesBackground;