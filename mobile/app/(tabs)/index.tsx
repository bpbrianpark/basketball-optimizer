import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  StatusBar,
} from 'react-native';
import { Link } from 'expo-router';

import { theme } from '@/constants/theme';

export default function HomeScreen() {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(32)).current;
  const ballAnim = useRef(new Animated.Value(0.7)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
      Animated.spring(slideAnim, { toValue: 0, tension: 80, friction: 10, useNativeDriver: true }),
      Animated.spring(ballAnim, { toValue: 1, tension: 55, friction: 8, delay: 100, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={theme.bg} />

      {/* Ambient background glow */}
      <View style={styles.bgGlow} />

      <Animated.View
        style={[
          styles.content,
          { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
        ]}>
        {/* Logo */}
        <Animated.View style={[styles.logoWrap, { transform: [{ scale: ballAnim }] }]}>
          <View style={styles.logoRing}>
            <Text style={styles.logoEmoji}>🏀</Text>
          </View>
        </Animated.View>

        {/* Title block */}
        <View style={styles.titleBlock}>
          <Text style={styles.eyebrow}>SHOT FORM</Text>
          <Text style={styles.title}>OPTIMIZER</Text>
          <View style={styles.accentBar} />
        </View>

        {/* Tagline */}
        <Text style={styles.tagline}>
          AI-powered pose analysis.{'\n'}Instant form feedback.
        </Text>

        {/* Stats row */}
        <View style={styles.statsRow}>
          {[
            { num: '4', label: 'JOINTS' },
            { num: '0–100', label: 'SCORE' },
            { num: 'AI', label: 'POWERED' },
          ].map(({ num, label }, i) => (
            <React.Fragment key={label}>
              {i > 0 && <View style={styles.statDivider} />}
              <View style={styles.stat}>
                <Text style={styles.statNum}>{num}</Text>
                <Text style={styles.statLabel}>{label}</Text>
              </View>
            </React.Fragment>
          ))}
        </View>

        {/* CTA */}
        <Link href="/(tabs)/capture" asChild>
          <TouchableOpacity style={styles.ctaButton} activeOpacity={0.82}>
            <Text style={styles.ctaText}>ANALYZE YOUR SHOT</Text>
            <Text style={styles.ctaArrow}>→</Text>
          </TouchableOpacity>
        </Link>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.bg,
    justifyContent: 'center',
    padding: 24,
  },
  bgGlow: {
    position: 'absolute',
    top: -100,
    right: -80,
    width: 280,
    height: 280,
    borderRadius: 140,
    backgroundColor: theme.accentDim,
    opacity: 0.7,
  },
  content: {
    alignItems: 'center',
  },
  logoWrap: {
    marginBottom: 32,
  },
  logoRing: {
    width: 96,
    height: 96,
    borderRadius: 48,
    borderWidth: 2,
    borderColor: theme.accent,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.surface,
  },
  logoEmoji: {
    fontSize: 44,
  },
  titleBlock: {
    alignItems: 'center',
    marginBottom: 16,
  },
  eyebrow: {
    fontSize: 12,
    letterSpacing: 6,
    color: theme.accent,
    fontFamily: 'SpaceMono',
    marginBottom: 4,
  },
  title: {
    fontSize: 52,
    fontWeight: '900',
    color: theme.textPrimary,
    letterSpacing: 4,
    lineHeight: 56,
  },
  accentBar: {
    width: 48,
    height: 3,
    backgroundColor: theme.accent,
    marginTop: 12,
    borderRadius: 2,
  },
  tagline: {
    fontSize: 14,
    color: theme.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 36,
    fontFamily: 'SpaceMono',
  },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.surface,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: theme.border,
    paddingVertical: 16,
    paddingHorizontal: 20,
    marginBottom: 40,
    width: '100%',
  },
  stat: {
    flex: 1,
    alignItems: 'center',
  },
  statNum: {
    fontSize: 17,
    fontWeight: '800',
    color: theme.accent,
    fontFamily: 'SpaceMono',
  },
  statLabel: {
    fontSize: 9,
    letterSpacing: 3,
    color: theme.textSecondary,
    marginTop: 3,
  },
  statDivider: {
    width: 1,
    height: 32,
    backgroundColor: theme.border,
  },
  ctaButton: {
    backgroundColor: theme.accent,
    paddingVertical: 18,
    paddingHorizontal: 32,
    borderRadius: 4,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    width: '100%',
    justifyContent: 'center',
  },
  ctaText: {
    fontSize: 13,
    fontWeight: '800',
    color: theme.bg,
    letterSpacing: 3,
  },
  ctaArrow: {
    fontSize: 18,
    color: theme.bg,
    fontWeight: '800',
  },
});
