import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  StatusBar,
} from 'react-native';
import { router } from 'expo-router';
import FontAwesome from '@expo/vector-icons/FontAwesome';

import { theme } from '@/constants/theme';

type ActionCardProps = {
  icon: React.ComponentProps<typeof FontAwesome>['name'];
  label: string;
  sublabel: string;
  onPress: () => void;
  delay: number;
};

function ActionCard({ icon, label, sublabel, onPress, delay }: ActionCardProps) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(20)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 400, delay, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 400, delay, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <Animated.View style={[styles.cardWrap, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}>
      <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.72}>
        <View style={styles.cardIconWrap}>
          <FontAwesome name={icon} size={28} color={theme.accent} />
        </View>
        <Text style={styles.cardLabel}>{label}</Text>
        <Text style={styles.cardSub}>{sublabel}</Text>
        <View style={styles.cardArrowWrap}>
          <Text style={styles.cardArrow}>→</Text>
        </View>
      </TouchableOpacity>
    </Animated.View>
  );
}

const TIPS = [
  'Film from the side at waist height',
  'Keep full body in frame',
  'Keep clip under 10 seconds',
];

export default function CaptureScreen() {
  const headerFade = useRef(new Animated.Value(0)).current;
  const tipsSlide = useRef(new Animated.Value(24)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(headerFade, { toValue: 1, duration: 500, useNativeDriver: true }),
      Animated.timing(tipsSlide, { toValue: 0, duration: 500, delay: 350, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={theme.bg} />

      {/* Header */}
      <Animated.View style={[styles.header, { opacity: headerFade }]}>
        <Text style={styles.eyebrow}>STEP 01</Text>
        <Text style={styles.title}>CAPTURE{'\n'}YOUR SHOT</Text>
        <View style={styles.accentBar} />
      </Animated.View>

      {/* Action cards */}
      <View style={styles.cardsRow}>
        <ActionCard
          icon="video-camera"
          label="RECORD"
          sublabel={'Live camera\n3–10 sec clip'}
          onPress={() => router.push('/(tabs)/results')}
          delay={120}
        />
        <View style={{ width: 12 }} />
        <ActionCard
          icon="photo"
          label="LIBRARY"
          sublabel={'Pick from\ncamera roll'}
          onPress={() => router.push('/(tabs)/results')}
          delay={240}
        />
      </View>

      {/* Tips */}
      <Animated.View
        style={[styles.tipsBlock, { opacity: headerFade, transform: [{ translateY: tipsSlide }] }]}>
        <Text style={styles.tipsHeader}>TIPS FOR BEST RESULTS</Text>
        {TIPS.map((tip, i) => (
          <View key={i} style={styles.tipRow}>
            <View style={styles.tipDot} />
            <Text style={styles.tipText}>{tip}</Text>
          </View>
        ))}
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.bg,
    padding: 24,
    paddingTop: 48,
  },
  header: {
    marginBottom: 32,
  },
  eyebrow: {
    fontSize: 11,
    letterSpacing: 5,
    color: theme.accent,
    fontFamily: 'SpaceMono',
    marginBottom: 6,
  },
  title: {
    fontSize: 40,
    fontWeight: '900',
    color: theme.textPrimary,
    letterSpacing: 2,
    lineHeight: 44,
  },
  accentBar: {
    width: 40,
    height: 3,
    backgroundColor: theme.accent,
    marginTop: 12,
    borderRadius: 2,
  },
  cardsRow: {
    flexDirection: 'row',
    marginBottom: 24,
  },
  cardWrap: {
    flex: 1,
  },
  card: {
    backgroundColor: theme.surface,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 4,
    padding: 20,
    minHeight: 180,
  },
  cardIconWrap: {
    width: 52,
    height: 52,
    borderRadius: 4,
    backgroundColor: theme.accentDim,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  cardLabel: {
    fontSize: 13,
    fontWeight: '800',
    color: theme.textPrimary,
    letterSpacing: 3,
    marginBottom: 6,
  },
  cardSub: {
    fontSize: 12,
    color: theme.textSecondary,
    lineHeight: 18,
    fontFamily: 'SpaceMono',
    flex: 1,
  },
  cardArrowWrap: {
    alignSelf: 'flex-end',
    paddingTop: 12,
  },
  cardArrow: {
    fontSize: 18,
    color: theme.accent,
    fontWeight: '600',
  },
  tipsBlock: {
    backgroundColor: theme.surface,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 4,
    padding: 20,
  },
  tipsHeader: {
    fontSize: 10,
    letterSpacing: 4,
    color: theme.textSecondary,
    marginBottom: 16,
  },
  tipRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  tipDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: theme.accent,
    marginRight: 12,
  },
  tipText: {
    fontSize: 13,
    color: theme.textPrimary,
    flex: 1,
    lineHeight: 20,
  },
});
