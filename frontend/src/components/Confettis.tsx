import { useEffect, useState } from 'react';
import Confetti from 'react-confetti';

export const ConfettiEffect = () => {
  const [isActive, setIsActive] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsActive(false);
    }, 5000); // Confetti s'arrête après 5 secondes

    return () => clearTimeout(timer); // Nettoyage du timer
  }, []);

  return isActive ? <Confetti /> : null;
};
