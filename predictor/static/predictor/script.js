document.addEventListener('DOMContentLoaded', function() {
  initRatings();
});

function initRatings() {
  const ratingContainers = document.querySelectorAll('.rating');
  
  ratingContainers.forEach(container => {
      const stars = container.querySelectorAll('i');
      const ratingText = container.querySelector('.rating-text');
      const hiddenInput = container.querySelector('input[type="hidden"]');

      stars.forEach(star => {
          // Hover effect
          star.addEventListener('mouseenter', () => {
              const value = parseFloat(star.dataset.value);
              updateStars(stars, value);
          });
          
          // Mouse leave - reset to selected value
          container.addEventListener('mouseleave', () => {
              const selectedValue = parseFloat(hiddenInput.value);
              updateStars(stars, selectedValue);
          });
          
          // Click to set rating
          star.addEventListener('click', () => {
              const value = parseFloat(star.dataset.value);
              hiddenInput.value = value;
              updateStars(stars, value);
              updateRatingText(ratingText, value);
              
              // Add animation
              star.classList.add('pulse');
              setTimeout(() => {
                  star.classList.remove('pulse');
              }, 300);
          });
      });
  });
}

function updateStars(stars, value) {
  stars.forEach(star => {
      const starValue = parseFloat(star.dataset.value);
      
      if (starValue <= value) {
          star.classList.remove('far');
          star.classList.add('fas');
      } else {
          star.classList.remove('fas');
          star.classList.add('far');
      }
      
      // Add half-star effect
      if (starValue - 0.5 <= value && starValue > value) {
          star.classList.add('fas', 'half-star');
      } else {
          star.classList.remove('half-star');
      }
  });
}

function updateRatingText(element, value) {
  if (value === 0) {
      element.textContent = 'Select rating';
  } else {
      element.textContent = `${value.toFixed(1)} / 5`;
  }
}

// Add pulse animation
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.2); }
      100% { transform: scale(1); }
  }
  .pulse {
      animation: pulse 0.3s;
  }
  .half-star::before {
      content: '\\f123'; /* FontAwesome half-star icon */
      position: absolute;
      width: 50%;
      overflow: hidden;
  }
`;
document.head.appendChild(style);
