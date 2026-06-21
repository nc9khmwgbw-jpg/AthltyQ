    <script>
        // Parallaxe GSAP pour les paragraphes et éléments de texte
        document.addEventListener("mousemove", (e) => {
            if (typeof gsap === 'undefined') return;
            
            const { clientX, clientY } = e;
            const centerX = window.innerWidth / 2;
            const centerY = window.innerHeight / 2;

            const moveX = (clientX - centerX) / centerX;
            const moveY = (clientY - centerY) / centerY;

            // Applique l'effet aux éléments textuels (titres, paragraphes, CTA)
            gsap.to(".hero-eyebrow-bar", { x: moveX * 15, y: moveY * 15, duration: 1, ease: "power2.out" });
            gsap.to(".hero-left-info", { x: moveX * -25, y: moveY * -25, duration: 1.5, ease: "power2.out" });
            gsap.to("#navLogoContainer", { x: moveX * 10, y: moveY * 10, duration: 1, ease: "power2.out" });
            gsap.to(".filter-title", { x: moveX * 5, duration: 2, ease: "power2.out" });
            
            // Si on a des p.cin-desc ou autres
            gsap.to(".cin-desc, p", { x: moveX * -10, y: moveY * -10, duration: 1.2, ease: "power2.out" });
        });
    </script>
