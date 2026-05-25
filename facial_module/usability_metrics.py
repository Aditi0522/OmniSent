##  Derives UX-relevant scores from 7-class emotion probabilities.
##  These are composite metrics, not new emotion classes.


class UsabilityMetricsCalculator:
    def compute(self, emotions: dict, confidence: float) -> dict:
        a = emotions.get('Angry', 0)
        d = emotions.get('Disgusted', 0)
        f = emotions.get('Fearful', 0)
        h = emotions.get('Happy', 0)
        n = emotions.get('Neutral', 0)
        sa = emotions.get('Sad', 0)
        su = emotions.get('Surprised', 0)

        frustration = a * 0.5 + d * 0.3 + sa * 0.2
        confusion = f * 0.4 + su * 0.3 + (1 - confidence) * 0.3
        engagement = h * 0.3 + su * 0.3 + (1 - n) * 0.4
        satisfaction = h * 0.6 + n * 0.3 * confidence + su * 0.1
        emotional_intensity = 1 - n
        boredom = (n * 0.5 + sa * 0.3 + d * 0.2) * (1 - emotional_intensity)
        stress = f * 0.5 + a * 0.3 + su * 0.2

        return {
            'frustration_index': round(float(min(1.0, frustration)), 4),
            'confusion_score': round(float(min(1.0, confusion)), 4),
            'engagement_level': round(float(min(1.0, engagement)), 4),
            'satisfaction_score': round(float(min(1.0, satisfaction)), 4),
            'boredom_score': round(float(min(1.0, boredom)), 4),
            'stress_level': round(float(min(1.0, stress)), 4),
        }

    def detect_events(self, emotions: dict, confidence: float,
                      prev_emotions: dict = None) -> list:
        """Detect discrete usability events worth flagging."""
        events = []
        m = self.compute(emotions, confidence)

        if m['frustration_index'] > 0.6:
            events.append({
                'type': 'frustration_spike',
                'severity': 'high' if m['frustration_index'] > 0.8 else 'medium',
                'detail': f"frustration={m['frustration_index']:.2f}",
            })

        if m['confusion_score'] > 0.5:
            events.append({
                'type': 'confusion_detected',
                'severity': 'high' if m['confusion_score'] > 0.7 else 'medium',
                'detail': f"confusion={m['confusion_score']:.2f}",
            })

        if emotions.get('Happy', 0) > 0.8 and confidence > 0.7:
            events.append({
                'type': 'delight_moment',
                'severity': 'positive',
                'detail': f"happy={emotions['Happy']:.2f}, conf={confidence:.2f}",
            })

        if prev_emotions:
            prev_dom = max(prev_emotions, key=prev_emotions.get)
            curr_dom = max(emotions, key=emotions.get)
            if prev_dom != curr_dom:
                shift = abs(emotions[curr_dom] - prev_emotions.get(curr_dom, 0))
                if shift > 0.4:
                    events.append({
                        'type': 'emotion_shift',
                        'from': prev_dom,
                        'to': curr_dom,
                        'magnitude': round(float(shift), 2),
                    })

        return events
